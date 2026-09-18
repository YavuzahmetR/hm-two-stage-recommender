import json
import random

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from prepare_two_tower import OUTPUT_DIR

SEED = 42
EMBEDDING_DIM = 64
BATCH_SIZE = 1024
NEGATIVES = 10
EPOCHS = 5
LEARNING_RATE = 0.001
TEMPERATURE = 0.1

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def encode_users(self, user_ids):
        vectors = self.user_tower(self.user_embedding(user_ids))
        return F.normalize(vectors, dim=-1)

    def encode_items(self, item_ids):
        vectors = self.item_tower(self.item_embedding(item_ids))
        return F.normalize(vectors, dim=-1)

    def forward(self, user_ids, item_ids):
        users_vectors = self.encode_users(user_ids)
        items_vectors = self.encode_items(item_ids)

        return (
            users_vectors.unsqueeze(1) * items_vectors
        ).sum(dim=-1) / TEMPERATURE


def sample_negatives(user_ids, n_items, known_keys, rng):
    negatives = rng.integers(
        0,
        n_items,
        size=(len(user_ids), NEGATIVES),
        dtype=np.int64,
    )

    for _ in range(100):
        keys = user_ids[:, None] * n_items + negatives
        positions = np.searchsorted(known_keys, keys)
        safe_positions = np.minimum(positions, len(known_keys) - 1)

        invalid = (
            (positions < len(known_keys))
            & (known_keys[safe_positions] == keys)
        )

        if not invalid.any():
            return negatives

        negatives[invalid] = rng.integers(
            0,
            n_items,
            size=int(invalid.sum()),
            dtype=np.int64,
        )

    raise RuntimeError("Could not sample valid negative items.")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run.")
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    rng = np.random.default_rng(SEED)

    metadata = json.loads(
        (OUTPUT_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    pairs = pl.read_parquet(OUTPUT_DIR / "interactions.parquet")

    n_users = metadata["n_users"]
    n_items = metadata["n_items"]

    user_ids = pairs["user_idx"].to_numpy().astype(np.int64)
    item_ids = pairs["item_idx"].to_numpy().astype(np.int64)

    if len(user_ids) != metadata["n_pairs"] or len(user_ids) == 0:
        raise ValueError("Interaction count does not match metadata.")

    if not (
        0 <= user_ids.min() <= user_ids.max() < n_users
        and 0 <= item_ids.min() <= item_ids.max() < n_items
    ):
        raise ValueError("Invalid user or item indices.")

    known_keys = np.unique(user_ids * n_items + item_ids)

    if len(known_keys) != len(user_ids):
        raise ValueError("Expected unique user-item pairs.")

    counts = np.bincount(user_ids, minlength=n_users)
    if (counts >= n_items).any():
        raise ValueError("A user has no available negative items.")

    model = TwoTower(
        n_users,
        n_items,
        EMBEDDING_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()


    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Positive pairs: {len(user_ids):,}")
    print(f"Negatives per positive: {NEGATIVES}")
    print(
        "Parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(user_ids))
        total_loss = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            batch_users = user_ids[batch_indices]
            positives = item_ids[batch_indices]

            negatives = sample_negatives(
                batch_users,
                n_items,
                known_keys,
                rng
            )

            candidate_items = np.concatenate(
                [positives[:, None], negatives],
                axis=1
            )

            user_tensor = torch.as_tensor(
                batch_users,
                dtype=torch.long,
                device=device
            )

            items_tensor = torch.as_tensor(
                candidate_items,
                dtype=torch.long,
                device=device
            )

            targets = torch.zeros(
                len(batch_indices),
                dtype=torch.long,
                device=device
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(user_tensor, items_tensor)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += loss.item() * len(batch_indices)
            seen += len(batch_indices)

        average_loss = total_loss / seen
        history.append({"epoch": epoch, "loss": average_loss})

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Training loss: {average_loss:.6f}"
        )

    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "n_users": n_users,
        "n_items": n_items,
        "embedding_dim": EMBEDDING_DIM,
        "temperature": TEMPERATURE,
        "epochs": EPOCHS,
        "seed": SEED,
        "as_of": metadata["as_of"],
        "lookback_days": metadata["lookback_days"],
        "negatives": NEGATIVES,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }

    torch.save(checkpoint, OUTPUT_DIR / "two_tower_v1.pt")
    pl.DataFrame(history).write_csv(
        OUTPUT_DIR / "training_history_v1.csv"
    )

    print("\nSaved two_tower_v1.pt and training_history_v1.csv")


if __name__ == "__main__":
    main()
