# deblur-unet — Sub-Roadmap (Level 2)

Features to try, in priority order.
The deblur-unet agent works through these one per run.

Focus areas rotate: Architecture → Augmentation → Loss → Regularization → Training dynamics

Status: TODO | IN_PROGRESS | DONE | BLOCKED

---

## Improvement items

- [ ] status: TODO
  id: dunet-focal-loss
  focus: Loss
  change: Replace cross-entropy with focal loss (gamma=2, alpha=0.25)
  rationale: partial_decode_rate is high but full_decode_rate is low; focal loss focuses on hard samples

- [ ] status: TODO
  id: dunet-se-blocks
  focus: Architecture
  change: Add squeeze-excitation blocks after each conv layer
  rationale: channel attention may help focus on barcode structure vs background noise
  depends_on: [dunet-focal-loss]

- [ ] status: TODO
  id: dunet-motion-blur-augmentation
  focus: Augmentation
  change: Add motion blur augmentation (kernel 5-15px, random angle) to training pipeline
  rationale: test set contains motion-blur samples not represented in training data

- [ ] status: TODO
  id: dunet-spatial-dropout
  focus: Regularization
  change: Add spatial dropout (rate=0.1) after each conv block
  rationale: model may be overfitting to specific degradation patterns

- [ ] status: TODO
  id: dunet-cosine-lr
  focus: Training dynamics
  change: Replace step LR decay with cosine annealing schedule
  rationale: step decay may be causing training instability near boundaries

---

## Completed

<!-- deblur-unet agent appends completed items here -->
