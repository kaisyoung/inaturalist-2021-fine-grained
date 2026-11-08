# Stage 4 Grad-CAM interpretation

## Evidence and scope

This qualitative analysis uses the frozen Stage 3 analysis model, `efficientnet_b0_pretrained_strong_aug`, without retraining or parameter updates. The deterministic selection contains 33 unique held-out test images: 12 highest-confidence correct predictions, 12 highest-confidence incorrect predictions that are not confusion-pair representatives, and 9 available directional representatives from the five most-confused class pairs. The run produced 54 maps: a predicted-class map for all 33 images and an additional true-class map for each of the 21 errors.

Grad-CAM was computed at EfficientNet-B0 layer `features.8` using the exact Stage 3 evaluation preprocessing. Each figure shows the EXIF-corrected original, the 224×224 model crop, the heatmap, and the heatmap mapped back to the original image. Pixels outside the center crop have zero overlay opacity.

## Observed patterns

- In most high-confidence correct examples, the strongest response overlaps the organism or a visibly diagnostic part: fruit for *Solanum pseudocapsicum*, flower structures for both orchid examples, the moth body and wings, the Indian Roller torso, the dragonfly body and wing bases, the puffball body, and the bellwort flower. This is consistent with object-focused evidence in these selected examples.
- The maps are spatially coarse. They often cover a whole flower head, bird torso, or compact organism rather than isolating a fine anatomical character. This is expected for a CAM derived from the final convolutional feature map and limits claims about which species-level character was decisive.
- Several high-confidence errors still focus on the correct foreground object. For image 2759395, the predicted *Solanum pseudocapsicum* map focuses on the red berry, while the true *Lycium barbarum* score is approximately zero and its target map is displaced toward the stem/background. For image 2758641, the predicted tick map is tightly centred on the actual shield bug, whereas the true-class map is weaker and partly offset. These are consistent with fine-grained shape/colour similarity rather than simple failure to find the organism.
- The high-confidence incorrect grid shows the same foreground-focused pattern for the blind-snake/groundsnake, butterfly, rabbit, aster, and small-flower examples. Attention to the relevant object is therefore not sufficient for correct 500-way species recognition.
- Image 2730603 is a conspicuous shortcut-like failure: a Red Junglefowl is predicted as white pasqueflower with probability 0.932. The predicted-class response is concentrated on the bird's pale, shaggy plumage, whose texture resembles a flower head; the true bird-class score is only 0.021 and its CAM is diffuse. This visual evidence is compatible with a texture cue dominating the correct object category, although Grad-CAM alone cannot prove causality.
- The confused-pair maps generally overlap the animals themselves for both heron directions, both damselfly directions, the honeyeaters, the warblers, and the vole. These pairs are visually and taxonomically close, supporting the interpretation that many residual errors are fine-grained within-group confusions rather than arbitrary cross-domain mistakes.
- Some maps include surrounding habitat or support surfaces together with the subject. The selected African Oystercatcher example, for instance, has activation spanning the bird/rock region. This may reflect contextual evidence, coarse spatial resolution, or both; the visualization cannot separate those explanations.

## Limitations

- Grad-CAM is a class-conditional localization heuristic, not a causal explanation or a pixel-accurate segmentation.
- The 33 examples are deliberately selected extremes and confusion representatives, so they are not an unbiased estimate of behaviour over all 5,000 test images.
- True-class CAMs for very low-scoring classes can be weak or unstable and should be interpreted as contrastive diagnostics, not evidence that the model has learned a reliable true-class representation.
- Heatmaps are normalized independently per image; colour intensity cannot be compared quantitatively across figures.
- Center-crop preprocessing can exclude peripheral content. The inverse mapping documents the evaluated crop but cannot recover evidence the model never received.

All quantitative failure claims remain grounded in the persisted prediction, per-class, confusion, and robustness tables; these visual observations are supplementary.
