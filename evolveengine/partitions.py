"""Request pools and role-scoped access to independent snapshot banks."""

import hashlib
import hmac
import random

from .schema import digest


ACCESS = {
    "search": frozenset("S"),
    "updater": frozenset(),
    "development": frozenset("SD"),
    "causal": frozenset("SC"),
    "final": frozenset("TQ"),
    "process": frozenset({"S_dev"}),
    "admission": frozenset({"Q_admit"}),
}
LENGTHS = (128, 512, 512, 2048)


class Partitions:
    def __init__(self, seed, salt=None, token_ids=range(32, 500), special_prefix=(), *,
                 final_salt=None, development_commitment=None, final_commitment=None):
        if type(seed) is not int:
            raise ValueError("Partitions need an integer seed")
        self.seed = seed
        self._development_salt = salt
        self._final_salt = final_salt
        self.development_commitment = _commitment(salt, development_commitment)
        self.final_commitment = _commitment(final_salt, final_commitment)
        if self.development_commitment and self.development_commitment == self.final_commitment:
            raise ValueError("Development and final pools require separate seal keys")
        self.token_ids = tuple(token_ids)
        self.special_prefix = tuple(special_prefix)
        if len(set(self.token_ids)) < 2 or len(self.special_prefix) >= min(LENGTHS):
            raise ValueError("Invalid tokenizer fixture")

    def block(self, role, pool, name, count):
        if pool not in ACCESS.get(role, ()):
            raise PermissionError(f"{role} cannot read partition {pool}")
        if type(count) is not int or count <= 0 or count % 4:
            raise ValueError("Block size must be divisible by four")
        key = f"v1/{self.seed}/{pool}/{name}".encode("utf-8")
        if pool in {"S", "S_dev"}:
            seed = hashlib.sha256(key).digest()
        else:
            salt = self._final_salt if pool in {"T", "Q"} else self._development_salt
            if salt is None:
                raise PermissionError(f"Key for partition {pool} was not loaded")
            seed = hmac.digest(salt, key, "sha256")
        generator = random.Random(int.from_bytes(seed, "big"))
        lengths = list(LENGTHS) * (count // 4)
        generator.shuffle(lengths)
        prompts = [
            [*self.special_prefix, *generator.choices(self.token_ids, k=length - len(self.special_prefix))]
            for length in lengths
        ]
        return {"partition": pool, "name": name, "prompts": prompts, "max_new_tokens": 128}

    def public_manifest(self):
        return {"seed": self.seed, "development_commitment": self.development_commitment,
                "final_commitment": self.final_commitment,
                "generator": "v1", "lengths": LENGTHS, "token_set_digest": digest(self.token_ids),
                "special_prefix": self.special_prefix}

    def development_view(self):
        return Partitions(self.seed, self._development_salt, self.token_ids, self.special_prefix,
                          final_commitment=self.final_commitment)

    def final_view(self):
        return Partitions(self.seed, token_ids=self.token_ids, special_prefix=self.special_prefix,
                          final_salt=self._final_salt, development_commitment=self.development_commitment,
                          final_commitment=self.final_commitment)


def _commitment(salt, expected):
    if expected is not None and not valid_digest(expected):
        raise ValueError("Seal commitment must be a lowercase SHA-256 digest")
    if salt is None:
        return expected
    if type(salt) is not bytes or len(salt) < 32:
        raise ValueError("Each seal key needs at least 32 private bytes")
    actual = hashlib.sha256(salt).hexdigest()
    if expected is not None and not hmac.compare_digest(expected, actual):
        raise ValueError("Seal key differs from its registered commitment")
    return actual


def valid_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def load_bank(journal, role, pool, bank):
    if pool not in {"D", "C"} or pool not in ACCESS.get(role, ()):
        raise PermissionError(f"{role} cannot read snapshot bank {pool}")
    snapshots = journal.get(f"bank_{pool}", bank)
    if any(not checkpoint["lineage"].startswith(f"bank/{pool}/") for checkpoint in snapshots):
        raise ValueError("Bank lineage does not match its partition")
    return snapshots
