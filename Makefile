CONFIG ?= work_dirs/bcss/classification/config.yaml
GPU ?= 0
LOG_DIR ?= logs
CHECKPOINT ?= work_dirs/bcss/classification/checkpoints/2025-11-24-07-07/best_cam.pth
SPLIT ?= test
OUT_DIR ?= outputs
IMAGES ?= TCGA-D8-A27F-DX1_xmin98787_ymin6725_MPP-0.2500+1.png

.PHONY: train
train:
	@STAMP=$$(date +"%Y%m%d-%H%M%S"); \
	mkdir -p $(LOG_DIR); \
	LOG=$(LOG_DIR)/train-$${STAMP}.log; \
	echo ">>> Logging to $$LOG"; \
	python main.py --config $(CONFIG) --gpu $(GPU) 2>&1 | tee $$LOG

.PHONY: eval
eval:
	@python visualization_utils/evaluate_performance.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --gpu $(GPU) --split $(SPLIT)

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
	@echo "  clean           Remove all files in LOG_DIR and OUT_DIR."
	@echo "  help            Show this help message."
	@echo "======================================================================"