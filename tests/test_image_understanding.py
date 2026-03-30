"""
Tests for the image understanding service (GPT-4o / GPT-5.4).

API tests  (pytest -m api):   verify the API works and returns useful descriptions.
Perf tests (pytest -m performance): verify description quality against product metadata.
"""
import base64
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GPT_MODEL = "gpt-4o"    # update to gpt-5.4 once available in your account


def _describe_image(client, image_path: Path) -> str:
    """
    Call GPT with an image and return the product description.
    This mirrors what the production Image Understanding tool will do.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are a product search assistant. "
                            "Describe this product image concisely for search purposes. "
                            "Include: product type, color, material (if visible), "
                            "style, and any notable features. "
                            "Be specific. Do not include prices or brand guesses."
                        ),
                    },
                ],
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# API Tests (U-A*)
# ---------------------------------------------------------------------------

class TestImageUnderstandingAPI:

    @pytest.mark.api
    def test_basic_connectivity(self, openai_client, sample_images):
        """U-A1: basic call returns non-empty description."""
        assert sample_images, "No sample images"
        desc = _describe_image(openai_client, sample_images[0])
        assert isinstance(desc, str)
        assert len(desc) > 20, f"Description too short: {desc!r}"

    @pytest.mark.api
    def test_jpeg_support(self, openai_client, sample_images):
        """U-A2: JPEG images are handled."""
        jpeg_images = [p for p in sample_images if p.suffix.lower() in (".jpg", ".jpeg")]
        assert jpeg_images, "No JPEG images in sample"
        desc = _describe_image(openai_client, jpeg_images[0])
        assert len(desc) > 20

    @pytest.mark.api
    def test_output_is_text(self, openai_client, sample_images):
        """U-A5: response is a non-empty string."""
        for img in sample_images[:3]:
            desc = _describe_image(openai_client, img)
            assert isinstance(desc, str)
            assert desc.strip(), "Got empty description"

    @pytest.mark.api
    def test_latency(self, openai_client, sample_images):
        """U-A6: p95 latency < 5s over 5 calls (allow for cold start)."""
        latencies = []
        for img in sample_images[:5]:
            t0 = time.perf_counter()
            _describe_image(openai_client, img)
            latencies.append(time.perf_counter() - t0)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        print(f"\nImage understanding p95 latency: {p95:.2f}s")
        assert p95 < 5.0, f"p95 latency {p95:.2f}s exceeds 5s threshold"

    @pytest.mark.api
    def test_sequential_calls_no_rate_error(self, openai_client, sample_images):
        """U-A7: 10 sequential calls all succeed."""
        errors = []
        for img in sample_images[:10]:
            try:
                desc = _describe_image(openai_client, img)
                assert len(desc) > 0
            except Exception as e:
                errors.append(str(e))
        assert not errors, f"Errors during sequential calls: {errors}"


# ---------------------------------------------------------------------------
# Performance Tests (U-P*)
# ---------------------------------------------------------------------------

class TestImageUnderstandingPerformance:

    # Common English color words to match against descriptions
    _ENGLISH_COLORS = {
        "red", "blue", "green", "white", "black", "brown", "gray", "grey",
        "pink", "yellow", "orange", "purple", "beige", "silver", "gold",
        "navy", "teal", "olive", "cream", "ivory", "coral", "tan", "dark", "light",
    }

    def _keywords_for_category(self, category: str) -> list[str]:
        """
        Build a keyword list for a given ABO product_type.
        Strategy: split the category name by underscore to get component words,
        then augment with a curated map for common types.
        """
        curated = {
            "SHOES": ["shoe", "sneaker", "boot", "sandal", "loafer", "heel", "footwear", "slipper"],
            "CHAIR": ["chair", "seat", "stool", "seating"],
            "SOFA": ["sofa", "couch", "sectional", "loveseat"],
            "RUG": ["rug", "carpet", "mat"],
            "GROCERY": ["food", "drink", "snack", "beverage", "sauce", "spice", "tea", "coffee",
                        "cake", "mix", "chocolate", "cereal", "nut", "pretzel", "pack", "box"],
            "HEALTH_PERSONAL_CARE": ["bottle", "cream", "lotion", "spray", "supplement",
                                     "vitamin", "tablet", "pill", "sunscreen", "allergy", "relief"],
            "SHIRT": ["shirt", "top", "blouse", "tee", "t-shirt"],
            "FINERING": ["ring", "jewelry", "jewellery", "band"],
            "FINEEARRING": ["earring", "jewelry", "jewellery", "stud", "hoop"],
            "WALL_ART": ["art", "print", "canvas", "poster", "painting", "picture", "frame"],
            "TABLE": ["table", "desk", "surface"],
            "HOME_FURNITURE_AND_DECOR": ["furniture", "decor", "lamp", "shelf", "cabinet", "vase", "pillow"],
            "CAMERA_BAGS_AND_CASES": ["bag", "case", "backpack", "pouch", "camera"],
            "POT_HOLDER": ["pot", "holder", "glove", "silicone", "handle", "oven"],
            "TOWEL_HOLDER": ["towel", "holder", "ring", "bar", "rack"],
            "ARTIFICIAL_PLANT": ["plant", "planter", "pot", "flower", "succulent", "vase", "hanging"],
            "PORTABLE_ELECTRONIC_DEVICE_MOUNT": ["mount", "holder", "strap", "camera", "action", "head"],
            "BOOT": ["boot", "shoe", "footwear"],
            "SANDAL": ["sandal", "shoe", "footwear", "flip"],
            "PET_SUPPLIES": ["pet", "dog", "cat", "collar", "leash", "bowl", "toy"],
            "OFFICE_PRODUCTS": ["pen", "pencil", "stapler", "tape", "binder", "organizer", "office"],
        }
        if category in curated:
            return curated[category]
        # Fallback: split underscore words, filter short ones
        words = [w.lower() for w in category.split("_") if len(w) > 2]
        return words or [category.lower()]

    def _extract_english_color(self, color_meta: str) -> str | None:
        """
        Extract a recognizable English color word from ABO color metadata.
        ABO color values can be multilingual; we look for known English color words.
        """
        if not color_meta:
            return None
        words = color_meta.lower().split()
        for word in words:
            if word in self._ENGLISH_COLORS:
                return word
        return None

    @pytest.mark.performance
    def test_category_identification(self, openai_client, sample_products, sample_images):
        """
        U-P1: description should contain at least one keyword matching the product category.
        Pass threshold: 70% (threshold relaxed from 85% because ABO has 400+ categories,
        many of which are niche and descriptions may use synonyms).
        """
        pairs = list(zip(sample_images[:10], sample_products[:10]))
        hits = 0
        misses = []

        for img_path, product in pairs:
            cat = product["category"]
            desc = _describe_image(openai_client, img_path).lower()
            keywords = self._keywords_for_category(cat)
            matched = any(kw in desc for kw in keywords)
            if matched:
                hits += 1
            else:
                misses.append({"category": cat, "description": desc[:120], "keywords": keywords})

        accuracy = hits / len(pairs)
        print(f"\nCategory ID accuracy: {accuracy:.0%} ({hits}/{len(pairs)})")
        for m in misses:
            print(f"  MISS cat={m['category']}  tried={m['keywords']}")
            print(f"       desc={m['description']!r}")

        assert accuracy >= 0.70, f"Category ID accuracy {accuracy:.0%} < 70%"

    @pytest.mark.performance
    def test_color_accuracy(self, openai_client, sample_products, sample_images):
        """
        U-P2: when product metadata contains a recognizable English color word,
        the GPT description should also mention that color.
        Only tests products whose color metadata contains a known English color.
        Pass threshold: 70%.
        """
        pairs = []
        for img, prod in zip(sample_images[:15], sample_products[:15]):
            color = self._extract_english_color(prod.get("color", ""))
            if color:
                pairs.append((img, prod, color))

        if len(pairs) < 3:
            pytest.skip(
                "Fewer than 3 products have English color metadata in this sample. "
                "ABO color fields are often multilingual. Skipping."
            )

        hits = 0
        misses = []
        for img_path, product, expected_color in pairs:
            desc = _describe_image(openai_client, img_path).lower()
            if expected_color in desc:
                hits += 1
            else:
                misses.append({"expected": expected_color, "raw_color": product["color"], "desc": desc[:100]})

        accuracy = hits / len(pairs)
        print(f"\nColor accuracy: {accuracy:.0%} ({hits}/{len(pairs)})")
        for m in misses:
            print(f"  MISS expected={m['expected']!r} (raw={m['raw_color']!r})  desc={m['desc']!r}")

        assert accuracy >= 0.70, f"Color accuracy {accuracy:.0%} < 70%"

    @pytest.mark.performance
    def test_attribute_completeness(self, openai_client, sample_images):
        """
        U-P3: description should mention at least 2 of {color, material, shape/style, type}.
        Mean across 10 images ≥ 2.0.
        """
        attribute_signals = {
            "color": ["red", "blue", "green", "white", "black", "brown", "gray", "grey",
                      "pink", "yellow", "orange", "purple", "beige", "silver", "gold",
                      "navy", "teal", "cream", "ivory", "dark", "light"],
            "material": ["leather", "cotton", "wool", "polyester", "wood", "metal",
                         "rubber", "fabric", "mesh", "canvas", "velvet", "suede",
                         "plastic", "steel", "silicone", "ceramic", "glass", "concrete"],
            "style": ["modern", "classic", "vintage", "casual", "formal", "sporty",
                      "minimalist", "traditional", "round", "square", "oval", "rectangular",
                      "cylindrical", "flat", "curved", "textured"],
            "type": ["shoe", "chair", "table", "sofa", "rug", "shirt", "jacket",
                     "bag", "ring", "necklace", "lamp", "shelf", "bottle", "box",
                     "backpack", "mount", "holder", "ring", "planter", "tablet",
                     "camera", "pot", "plant", "towel"],
        }

        counts = []
        for img in sample_images[:10]:
            desc = _describe_image(openai_client, img).lower()
            found = sum(
                1 for signals in attribute_signals.values()
                if any(s in desc for s in signals)
            )
            counts.append(found)
            print(f"  {img.name}: {found}/4  {desc[:80]!r}")

        mean_count = sum(counts) / len(counts)
        print(f"\nMean attribute completeness: {mean_count:.2f}/4")
        assert mean_count >= 2.0, f"Mean attribute count {mean_count:.2f} < 2.0"

    @pytest.mark.performance
    def test_searchability(self, openai_client, sample_products, sample_images):
        """
        U-P4: embed the GPT description and search — the source product should
        appear in top 5 results.
        This test requires the embedding client to be available.
        """
        from embed import EmbeddingClient
        import os

        if not os.environ.get("VOLCANO_AK") and not os.environ.get("ARK_EMBEDDING_ENDPOINT"):
            pytest.skip("Embedding credentials not set — cannot run searchability test")

        client = EmbeddingClient.from_env()

        # Pre-compute description vectors for all sample products
        descs = [p["description"][:400] for p in sample_products[:10]]
        catalog_vecs = client.embed_texts(descs)

        hits = 0
        for i, (img, product) in enumerate(zip(sample_images[:10], sample_products[:10])):
            # Get GPT description and embed it
            gpt_desc = _describe_image(openai_client, img)
            query_vec = client.embed_text(gpt_desc)

            # Rank catalog by cosine sim
            sims = [
                (j, client.cosine_similarity(query_vec, catalog_vecs[j]))
                for j in range(len(catalog_vecs))
            ]
            top5_ids = [j for j, _ in sorted(sims, key=lambda x: -x[1])[:5]]

            if i in top5_ids:
                hits += 1

        p_at_5 = hits / min(10, len(sample_images))
        print(f"\nSearchability P@5: {p_at_5:.0%} ({hits}/10)")
        assert p_at_5 >= 0.70, f"P@5 {p_at_5:.0%} < 70% threshold"
