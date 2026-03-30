"""
Shared data models for the preprocessing pipeline.
Used by both the test pipeline (500 products) and production pipeline (full catalog).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Product:
    item_id: str
    name: str
    description: str
    category: str
    brand: str
    color: str
    material: str
    image_path: Path          # absolute local path to the resolved image file
    image_url: str            # relative path used as the serving URL key
    web_url: str
    keywords: list[str] = field(default_factory=list)
    bullet_points: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "item_id": self.item_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "brand": self.brand,
            "color": self.color,
            "material": self.material,
            "image_path": str(self.image_path),
            "image_url": self.image_url,
            "web_url": self.web_url,
            "keywords": self.keywords,
            "bullet_points": self.bullet_points,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Product":
        return cls(
            item_id=d["item_id"],
            name=d["name"],
            description=d["description"],
            category=d["category"],
            brand=d["brand"],
            color=d["color"],
            material=d["material"],
            image_path=Path(d["image_path"]),
            image_url=d["image_url"],
            web_url=d["web_url"],
            keywords=d.get("keywords", []),
            bullet_points=d.get("bullet_points", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class EmbeddedProduct(Product):
    image_vector: list[float] = field(default_factory=list)
    description_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["image_vector"] = self.image_vector
        d["description_vector"] = self.description_vector
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddedProduct":
        base = Product.from_dict(d)
        return cls(
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            image_vector=d.get("image_vector", []),
            description_vector=d.get("description_vector", []),
        )
