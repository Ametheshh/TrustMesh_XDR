import json
import unittest
from pathlib import Path


ARTIFACT_DIR = Path("docs/threat-intelligence")


class ThreatIntelligenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = json.loads((ARTIFACT_DIR / "ciciomt2024_arp_spoofing.json").read_text())
        cls.stix = json.loads((ARTIFACT_DIR / "ciciomt2024_arp_spoofing.stix.json").read_text())

    def test_arp_spoofing_maps_to_approved_technique(self):
        self.assertEqual(self.mapping["source"]["dataset_id"], "ciciomt2024_wifi_mqtt")
        self.assertEqual(self.mapping["source"]["source_label"], "ARP_Spoofing")
        self.assertEqual(self.mapping["source"]["normalized_label"], "ciciomt2024_wifi_mqtt:arp_spoofing")
        self.assertEqual(self.mapping["mapping"]["attack_technique_id"], "T1557.002")
        self.assertEqual(self.mapping["mapping"]["attack_technique_name"], "ARP Cache Poisoning")

    def test_provenance_and_evidence_qualification_are_preserved(self):
        mapping = self.mapping["mapping"]
        self.assertEqual(mapping["evidence_basis"], "dataset_label")
        self.assertIn("no packet-level evidence", mapping["evidence_limitation"])
        self.assertIn("label-level", mapping["confidence"])
        self.assertEqual(self.mapping["sigma"]["status"], "not_generated")
        self.assertIn("No defensible event condition", self.mapping["sigma"]["limitation"])

    def test_stix_bundle_has_valid_required_structure_and_mapping_refs(self):
        bundle = self.stix
        self.assertEqual(bundle["type"], "bundle")
        self.assertEqual(bundle["spec_version"], "2.1")
        objects = {obj["type"]: obj for obj in bundle["objects"]}
        attack_pattern = objects["attack-pattern"]
        note = objects["note"]
        self.assertEqual(attack_pattern["name"], "ARP Cache Poisoning")
        self.assertEqual(attack_pattern["external_references"][0]["external_id"], "T1557.002")
        self.assertEqual(note["object_refs"], [attack_pattern["id"]])
        for stix_object in bundle["objects"]:
            self.assertIn(stix_object["type"], stix_object["id"])
            self.assertEqual(stix_object["spec_version"], "2.1")
            self.assertTrue(stix_object["created"])
            self.assertTrue(stix_object["modified"])
        self.assertIn("source label=ARP_Spoofing", note["content"])
        self.assertIn("Evidence basis: dataset_label", note["content"])
        self.assertIn("no packet-level evidence", note["content"])


if __name__ == "__main__":
    unittest.main()
