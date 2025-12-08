CONFIG ?= work_dirs/bcss/classification/config.yaml
GPU ?= 0
LOG_DIR ?= logs
CHECKPOINT ?= work_dirs/bcss/classification/checkpoints/2025-12-06-19-55/best_cam.pth
CHECKPOINT_DISTILLED ?= $(CHECKPOINT)
CHECKPOINT_BASELINE ?= checkpoints/conch/pytorch_model.bin
SPLIT ?= test
OUT_DIR ?= outputs
ANALYSIS_OUT ?= figures/analysis
IMAGES ?= TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+1.png

.PHONY: train
train:
	@STAMP=$$(date +"%Y%m%d-%H%M%S"); \
	mkdir -p $(LOG_DIR); \
	LOG=$(LOG_DIR)/train-$${STAMP}.log; \
	echo ">>> Logging to $$LOG"; \
	python main.py --config $(CONFIG) --gpu $(GPU) | tee $$LOG

.PHONY: eval
eval:
	@STAMP=$$(date +"%Y%m%d-%H%M%S"); \
	mkdir -p $(LOG_DIR); \
	LOG=$(LOG_DIR)/eval-$${STAMP}.log; \
	echo ">>> Logging to $$LOG"; \
	python visualization_utils/evaluate_performance.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --gpu $(GPU) --split $(SPLIT) 2>&1 | tee $$LOG

.PHONY: cam
cam:
	@OUT_ARG=""; \
	if [ -n "$(OUT_DIR)" ]; then OUT_ARG="--out-dir $(OUT_DIR)"; fi; \
	python visualization_utils/generate_cam.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --gpu $(GPU) --split $(SPLIT) $$OUT_ARG

.PHONY: proto
proto:
	@if [ -z "$(IMAGES)" ]; then echo "IMAGES is required (space-separated file names)"; exit 1; fi; \
	OUT_ARG=""; \
	if [ -n "$(OUT_DIR)" ]; then OUT_ARG="--out-dir $(OUT_DIR)"; fi; \
	python visualization_utils/visualize_prototypes.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --gpu $(GPU) --split $(SPLIT) --images $(IMAGES) $$OUT_ARG

.PHONY: structure
structure:
	@if [ -z "$(CHECKPOINT_DISTILLED)" ]; then echo "CHECKPOINT_DISTILLED is required"; exit 1; fi; \
	BASELINE_ARG=""; \
	if [ -n "$(CHECKPOINT_BASELINE)" ]; then BASELINE_ARG="--checkpoint-baseline $(CHECKPOINT_BASELINE)"; fi; \
	python visualization_utils/structure_analysis.py --config $(CONFIG) --checkpoint-distilled $(CHECKPOINT_DISTILLED) $$BASELINE_ARG --gpu $(GPU) --split $(SPLIT) --output-dir $(ANALYSIS_OUT)


.PHONY: clean
clean:
	@echo "Cleaning up log and output directories..."
	@rm -rf $(LOG_DIR)/*
	@rm -rf $(OUT_DIR)/*
	@echo "Cleanup complete."

.PHONY: help
help:
	@echo "======================================================================"
	@echo "                        TEFU Project Makefile                         "
	@echo "======================================================================"
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@echo "  train           Start the training process."
	@echo "                  Logs output to LOG_DIR with a timestamp."
	@echo "                  Variables:"
	@echo "                    CONFIG   : Path to config file (default: $(CONFIG))"
	@echo "                    GPU      : GPU ID to use (default: $(GPU))"
	@echo "                    LOG_DIR  : Directory for logs (default: $(LOG_DIR))"
	@echo ""
	@echo "  eval            Evaluate model performance metrics (F1, Accuracy, etc.)."
	@echo "                  Runs visualization_utils/evaluate_performance.py."
	@echo "                  Variables:"
	@echo "                    CONFIG   : Path to config file"
	@echo "                    CHECKPOINT: Path to model weights (default: $(CHECKPOINT))"
	@echo "                    GPU      : GPU ID"
	@echo "                    SPLIT    : Dataset split to use (default: $(SPLIT))"
	@echo ""
	@echo "  cam             Generate Class Activation Maps (CAM) visualizations."
	@echo "                  Runs visualization_utils/generate_cam.py."
	@echo "                  Variables:"
	@echo "                    CONFIG, CHECKPOINT, GPU, SPLIT"
	@echo "                    OUT_DIR  : Optional output directory for images"
	@echo ""
	@echo "  proto           Visualize specific prototype activations on images."
	@echo "                  Runs visualization_utils/visualize_prototypes.py."
	@echo "                  Variables:"
	@echo "                    IMAGES   : (Required) Space-separated filenames"
	@echo "                    CONFIG, CHECKPOINT, GPU, SPLIT, OUT_DIR"
	@echo ""
	@echo "  structure       Run layer-similarity + ERF analysis (baseline vs distilled)."
	@echo "                  Runs visualization_utils/structure_analysis.py."
	@echo "                  Variables:"
	@echo "                    CHECKPOINT_DISTILLED : Distilled checkpoint (required, default: $(CHECKPOINT_DISTILLED))"
	@echo "                    CHECKPOINT_BASELINE  : Optional baseline checkpoint (empty to skip)"
	@echo "                    CONFIG, GPU, SPLIT, ANALYSIS_OUT"
	@echo ""
	@echo "  clean           Remove all files in LOG_DIR and OUT_DIR."
	@echo "  help            Show this help message."
	@echo "======================================================================"
