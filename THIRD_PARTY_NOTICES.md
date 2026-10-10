# Third-party sources and licenses

- RoboSynChallenge source/configs/assets: EDEM-AI/RoboSynChallenge, commit 8818c82df5fa2522809b8105e7018950dcc9a9a8. Upstream Apache-2.0 LICENSE is included at the package root. Upstream evaluator and success criteria are not modified. https://github.com/EDEM-AI/RoboSynChallenge
- LeRobot: Hugging Face, runtime based on commit 8fff0fde7c79f23a93d845d1a50e985de01f8b8a, version 0.4.4; complete source used by the training environment is included under vendor/lerobot with its license. Local runtime modifications, if any, are preserved rather than replaced by an unverified new release. https://github.com/huggingface/lerobot
- Shared VLM/tokenizer: HuggingFaceTB/SmolVLM2-500M-Video-Instruct, snapshot 7b375e1b73b11138ff12fe22c8f2822d8fe03467. Model card and licensing information: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- Seven task checkpoints: RoboSynChallenge/SmolVLA_sim_*; exact repository revisions and LFS SHA256 are recorded in provenance/official_models.json. These remain official checkpoints, not team-trained weights. Model repository metadata did not supply a separate license file; organizer publication is not an independent legal clearance for arbitrary redistribution. This bundle is for the authorized competition teammate handoff; preserve attribution and ask the organizer if later publishing elsewhere.
- Three team variants: drawer recovery5k (65/100), basket Contact-v2 (54/100), pipette fixed9mm 50k (49/100). Their segmentation networks predict masks from RGB at deployment. Historical scores use different seeds/protocol dates and are not formal hidden-test results.

Submission compatibility change to vendored LeRobot: `configs/policies.py` accepts the nullable `pretrained_revision` metadata present in official checkpoints. This does not change tensor weights, inference settings, or action normalization.

Official submission rules: https://robosyn-bench.net/#/faq . This package does not constitute organizer approval, submission acceptance, or a guarantee of private evaluation performance.
