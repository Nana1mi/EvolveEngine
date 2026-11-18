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
    "admission": frozenset({"Q_admit"}),
}
LENGTHS = (128, 512, 512, 2048)


class Partitions:
    def __init__(self, seed, salt, token_ids=range(32, 500), special_prefix=()):
        if type(seed) is not int or not isinstance(salt, bytes) or len(salt) < 32:
            raise ValueError("Partitions need an integer seed and at least 32 private salt bytes")
        self.seed = seed
        self.salt = salt
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
        seed = (hashlib.sha256(key).digest() if pool == "S"
                else hmac.digest(self.salt, key, "sha256"))
        generator = random.Random(int.from_bytes(seed, "big"))
        lengths = list(LENGTHS) * (count // 4)
        generator.shuffle(lengths)
        prompts = [
            [*self.special_prefix, *generator.choices(self.token_ids, k=length - len(self.special_prefix))]
            for length in lengths
        ]
        return {"partition": pool, "name": name, "prompts": prompts, "max_new_tokens": 128}

    def public_manifest(self):
        return {"seed": self.seed, "seal_commitment": hashlib.sha256(self.salt).hexdigest(),
                "generator": "v1", "lengths": LENGTHS, "token_set_digest": digest(self.token_ids),
                "special_prefix": self.special_prefix}


def load_bank(journal, role, pool, bank):
    if pool not in {"D", "C"} or pool not in ACCESS.get(role, ()):
        raise PermissionError(f"{role} cannot read snapshot bank {pool}")
    snapshots = journal.get(f"bank_{pool}", bank)
    if any(not checkpoint["lineage"].startswith(f"bank/{pool}/") for checkpoint in snapshots):
        raise ValueError("Bank lineage does not match its partition")
    return snapshots
