"""
Structured design spaces.

A design space declares, for each design option, the values it may take. This
module turns such a declaration into the four operations that every other part
of the system needs:

    sample()      - draw a random valid design (GA initialisation, fallbacks)
    mutate()      - perturb a design (GA mutation operator)
    crossover()   - recombine two designs (GA crossover operator)
    encode()      - design -> fixed-length numeric vector (meta-learner input)
    decode()      - fixed-length numeric vector -> design (design prediction)

Keeping all five in one place is what guarantees the meta-learner, the GA and
the application all agree on what a design *is*.

Gene specifications
-------------------
    {'type': 'categorical', 'options': [...]}
    {'type': 'continuous', 'min': lo, 'max': hi, 'log': False, 'disabled_value': None}
    {'type': 'integer', 'min': lo, 'max': hi}

A ``block_list`` option holds a variable-length list of repeated gene groups.
It is what gives the CNN design space its variable-length chromosome (a design
with three convolutional layers is longer than one with two), as described in
Appendix A.3 of the thesis:

    {'type': 'block_list', 'min_blocks': 1, 'max_blocks': 6, 'genes': {...}}

Encoding is always *fixed length* even for variable-length designs: every
block_list is encoded as ``max_blocks`` slots, each carrying a presence flag
plus its genes, with absent slots zero-filled. Fixed-length encodings are a
hard requirement for the meta-learners, which cannot fit ragged input.
"""

from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np


def _as_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


class DesignSpace:
    """A structured, sampleable, encodable design space."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.option_names: List[str] = list(spec.keys())
        self._layout = self._build_layout()
        self.encoding_length = sum(width for _, width in self._layout)

    # ------------------------------------------------------------------
    # Layout / encoding bookkeeping
    # ------------------------------------------------------------------

    def _gene_width(self, gene: Dict[str, Any]) -> int:
        """Number of encoded dimensions for a single gene."""
        gene_type = gene['type']
        if gene_type == 'categorical':
            return len(gene['options'])
        if gene_type == 'continuous':
            # one value, plus an 'enabled' flag when the gene can be switched off
            return 2 if gene.get('disabled_value') is not None else 1
        if gene_type == 'integer':
            return 1
        raise ValueError(f"Unknown gene type: {gene_type}")

    def _build_layout(self) -> List[Tuple[str, int]]:
        """Compute (option_name, encoded_width) for every top-level option."""
        layout = []
        for name in self.option_names:
            option = self.spec[name]
            if option['type'] == 'block_list':
                per_block = 1 + sum(
                    self._gene_width(g) for g in option['genes'].values()
                )
                layout.append((name, per_block * option['max_blocks']))
            else:
                layout.append((name, self._gene_width(option)))
        return layout

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_gene(self, gene: Dict[str, Any], rng: np.random.Generator) -> Any:
        """Draw one random value for a single gene.

        Selection is by *index* rather than ``rng.choice(options)``: numpy
        coerces a heterogeneous option list to a common dtype, which silently
        turns numeric options into strings.
        """
        gene_type = gene['type']

        if gene_type == 'categorical':
            options = gene['options']
            return options[int(rng.integers(len(options)))]

        if gene_type == 'continuous':
            disabled_value = gene.get('disabled_value')
            if disabled_value is not None:
                if rng.random() < gene.get('disabled_prob', 0.5):
                    return disabled_value
            lo, hi = gene['min'], gene['max']
            if gene.get('log', False):
                return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            return float(rng.uniform(lo, hi))

        if gene_type == 'integer':
            return int(rng.integers(gene['min'], gene['max'] + 1))

        raise ValueError(f"Unknown gene type: {gene_type}")

    def _sample_block(self, genes: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
        return {name: self._sample_gene(gene, rng) for name, gene in genes.items()}

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Draw a uniformly random valid design."""
        rng = _as_rng(rng)
        design: Dict[str, Any] = {}

        for name in self.option_names:
            option = self.spec[name]
            if option['type'] == 'block_list':
                num_blocks = int(rng.integers(option['min_blocks'], option['max_blocks'] + 1))
                design[name] = [
                    self._sample_block(option['genes'], rng) for _ in range(num_blocks)
                ]
            else:
                design[name] = self._sample_gene(option, rng)

        return design

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def mutate(
        self,
        design: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
        gene_rate: float = 0.1,
        structure_rate: float = 0.15,
    ) -> Dict[str, Any]:
        """Return a mutated copy of ``design``.

        Two kinds of mutation are applied:
          * gene mutation - each gene is resampled with probability ``gene_rate``
          * structural mutation - a block_list grows or shrinks by one block
            with probability ``structure_rate``, which is how the GA explores
            variable-length chromosomes (number of layers).
        """
        rng = _as_rng(rng)
        mutant = copy.deepcopy(design)

        for name in self.option_names:
            option = self.spec[name]

            if option['type'] == 'block_list':
                blocks = mutant.get(name, [])

                # Structural mutation: add or remove a block.
                if rng.random() < structure_rate:
                    grow = rng.random() < 0.5
                    if grow and len(blocks) < option['max_blocks']:
                        insert_at = int(rng.integers(len(blocks) + 1))
                        blocks.insert(insert_at, self._sample_block(option['genes'], rng))
                    elif not grow and len(blocks) > option['min_blocks']:
                        blocks.pop(int(rng.integers(len(blocks))))

                # Gene mutation within each surviving block.
                for block in blocks:
                    for gene_name, gene in option['genes'].items():
                        if rng.random() < gene_rate:
                            block[gene_name] = self._sample_gene(gene, rng)

                mutant[name] = blocks
            else:
                if rng.random() < gene_rate:
                    mutant[name] = self._sample_gene(option, rng)

        return mutant

    def crossover(
        self,
        parent_a: Dict[str, Any],
        parent_b: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Recombine two designs into one child.

        Scalar options are inherited uniformly from either parent. A block_list
        is spliced at a random cut point (one-point crossover over the
        variable-length part), so the child can inherit a different number of
        layers than either parent.
        """
        rng = _as_rng(rng)
        child: Dict[str, Any] = {}

        for name in self.option_names:
            option = self.spec[name]

            if option['type'] == 'block_list':
                blocks_a = parent_a.get(name, [])
                blocks_b = parent_b.get(name, [])
                cut_a = int(rng.integers(len(blocks_a) + 1)) if blocks_a else 0
                cut_b = int(rng.integers(len(blocks_b) + 1)) if blocks_b else 0
                spliced = copy.deepcopy(blocks_a[:cut_a]) + copy.deepcopy(blocks_b[cut_b:])

                # Repair the splice so it respects the declared block bounds.
                while len(spliced) > option['max_blocks']:
                    spliced.pop()
                while len(spliced) < option['min_blocks']:
                    spliced.append(self._sample_block(option['genes'], rng))

                child[name] = spliced
            else:
                source = parent_a if rng.random() < 0.5 else parent_b
                child[name] = copy.deepcopy(source[name])

        return child

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _encode_gene(self, gene: Dict[str, Any], value: Any) -> List[float]:
        gene_type = gene['type']

        if gene_type == 'categorical':
            options = gene['options']
            return [1.0 if option == value else 0.0 for option in options]

        if gene_type == 'continuous':
            disabled_value = gene.get('disabled_value')
            lo, hi = gene['min'], gene['max']

            if disabled_value is not None and value == disabled_value:
                return [0.0, 0.0]  # disabled flag off, value irrelevant

            value = float(min(max(float(value), lo), hi))
            if gene.get('log', False):
                span = np.log(hi) - np.log(lo)
                norm = (np.log(value) - np.log(lo)) / span if span > 0 else 0.5
            else:
                span = hi - lo
                norm = (value - lo) / span if span > 0 else 0.5

            return [1.0, float(norm)] if disabled_value is not None else [float(norm)]

        if gene_type == 'integer':
            lo, hi = gene['min'], gene['max']
            span = hi - lo
            return [float((float(value) - lo) / span) if span > 0 else 0.5]

        raise ValueError(f"Unknown gene type: {gene_type}")

    def encode(self, design: Dict[str, Any]) -> np.ndarray:
        """Encode a design as a fixed-length float vector."""
        encoded: List[float] = []

        for name in self.option_names:
            option = self.spec[name]

            if option['type'] == 'block_list':
                blocks = design.get(name, [])
                for slot in range(option['max_blocks']):
                    if slot < len(blocks):
                        encoded.append(1.0)  # presence flag
                        for gene_name, gene in option['genes'].items():
                            encoded.extend(self._encode_gene(gene, blocks[slot][gene_name]))
                    else:
                        width = 1 + sum(self._gene_width(g) for g in option['genes'].values())
                        encoded.extend([0.0] * width)
            else:
                encoded.extend(self._encode_gene(option, design[name]))

        return np.asarray(encoded, dtype=np.float32)

    def _decode_gene(self, gene: Dict[str, Any], values: np.ndarray) -> Any:
        gene_type = gene['type']

        if gene_type == 'categorical':
            options = gene['options']
            return options[int(np.argmax(values))]

        if gene_type == 'continuous':
            disabled_value = gene.get('disabled_value')
            lo, hi = gene['min'], gene['max']

            if disabled_value is not None:
                if values[0] < 0.5:
                    return disabled_value
                norm = float(values[1])
            else:
                norm = float(values[0])

            norm = min(max(norm, 0.0), 1.0)
            if gene.get('log', False):
                value = float(np.exp(np.log(lo) + norm * (np.log(hi) - np.log(lo))))
            else:
                value = float(lo + norm * (hi - lo))
            # Clamp: exp/log round-tripping can land a hair outside the bound.
            return float(min(max(value, lo), hi))

        if gene_type == 'integer':
            lo, hi = gene['min'], gene['max']
            norm = min(max(float(values[0]), 0.0), 1.0)
            return int(round(lo + norm * (hi - lo)))

        raise ValueError(f"Unknown gene type: {gene_type}")

    def decode(self, vector: np.ndarray) -> Dict[str, Any]:
        """Decode a fixed-length float vector back into a valid design.

        Meta-learner output is an unconstrained regression, so every field is
        clamped back into the declared space. The result is always a design the
        application can actually build.
        """
        vector = np.asarray(vector, dtype=np.float32).ravel()
        if vector.size != self.encoding_length:
            raise ValueError(
                f"Expected an encoding of length {self.encoding_length}, got {vector.size}"
            )

        design: Dict[str, Any] = {}
        cursor = 0

        for name in self.option_names:
            option = self.spec[name]

            if option['type'] == 'block_list':
                genes = option['genes']
                per_block = 1 + sum(self._gene_width(g) for g in genes.values())

                slots = []
                presence = []
                for slot in range(option['max_blocks']):
                    chunk = vector[cursor:cursor + per_block]
                    cursor += per_block
                    presence.append(float(chunk[0]))

                    offset = 1
                    block = {}
                    for gene_name, gene in genes.items():
                        width = self._gene_width(gene)
                        block[gene_name] = self._decode_gene(gene, chunk[offset:offset + width])
                        offset += width
                    slots.append(block)

                # The number of blocks is the rounded sum of the presence
                # flags, clamped to the declared bounds; the highest-scoring
                # slots are the ones kept.
                num_blocks = int(round(sum(presence)))
                num_blocks = min(max(num_blocks, option['min_blocks']), option['max_blocks'])
                keep = sorted(np.argsort(presence)[::-1][:num_blocks])
                design[name] = [slots[i] for i in keep]
            else:
                width = self._gene_width(option)
                design[name] = self._decode_gene(option, vector[cursor:cursor + width])
                cursor += width

        return design

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, design: Dict[str, Any]) -> bool:
        """Check that a design conforms to this space."""
        if not isinstance(design, dict):
            return False

        for name in self.option_names:
            if name not in design:
                return False
            option = self.spec[name]

            if option['type'] == 'block_list':
                blocks = design[name]
                if not isinstance(blocks, list):
                    return False
                if not option['min_blocks'] <= len(blocks) <= option['max_blocks']:
                    return False
                for block in blocks:
                    for gene_name, gene in option['genes'].items():
                        if gene_name not in block:
                            return False
                        if not self._validate_gene(gene, block[gene_name]):
                            return False
            elif not self._validate_gene(option, design[name]):
                return False

        return True

    def _validate_gene(self, gene: Dict[str, Any], value: Any) -> bool:
        gene_type = gene['type']

        if gene_type == 'categorical':
            return value in gene['options']

        if gene_type == 'continuous':
            if gene.get('disabled_value') is not None and value == gene['disabled_value']:
                return True
            try:
                return gene['min'] <= float(value) <= gene['max']
            except (TypeError, ValueError):
                return False

        if gene_type == 'integer':
            try:
                return gene['min'] <= int(value) <= gene['max']
            except (TypeError, ValueError):
                return False

        return False


def space_from_legacy(legacy_space: Dict[str, Any]) -> DesignSpace:
    """Build a :class:`DesignSpace` from the old flat ``{name: [values]}`` format.

    The older applications declare their space as a list of options per design
    option, with ``[lo, hi, 'continuous']`` marking a continuous range. This
    adapter lets those applications keep their existing declaration while still
    getting correct sampling and fixed-length encoding.
    """
    spec: Dict[str, Any] = {}

    for name, values in legacy_space.items():
        if isinstance(values, dict):
            spec[name] = values  # already in structured form
        elif (
            isinstance(values, (list, tuple))
            and len(values) == 3
            and values[2] == 'continuous'
        ):
            spec[name] = {'type': 'continuous', 'min': values[0], 'max': values[1]}
        else:
            spec[name] = {'type': 'categorical', 'options': list(values)}

    return DesignSpace(spec)
