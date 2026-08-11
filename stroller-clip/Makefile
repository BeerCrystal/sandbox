SCAD  ?= openscad
OUT   ?= build

# Your own cup holder STL, welded on by the `fused` part:
#     make CUP=vendor/PriamCupHolderV3.stl
# Leave unset to use the built-in ring.
CUP   ?=
CUPD   = $(if $(CUP),-D 'external_cup="$(abspath $(CUP))"',)
PARTS  = caddy fused cupholder testfit

# xvfb-run is only needed for the PNG previews; STL export is headless.
XVFB  ?= $(shell command -v xvfb-run 2>/dev/null) $(if $(shell command -v xvfb-run 2>/dev/null),-a,)

.PHONY: all check preview clean

all: $(PARTS:%=$(OUT)/%.stl) $(OUT)/gauge.stl $(OUT)/corner_gauge.stl

$(OUT):
	mkdir -p $(OUT)

$(OUT)/gauge.stl: gauge.scad | $(OUT)
	$(SCAD) --export-format=binstl -o $@ $<

$(OUT)/corner_gauge.stl: corner_gauge.scad | $(OUT)
	$(SCAD) --export-format=binstl -o $@ $<

$(OUT)/%.stl: stroller_clip.scad | $(OUT)
	$(SCAD) --export-format=binstl -o $@ -D 'part="$*"' $(CUPD) $<

# Clearance checks. Each must produce an empty result; anything solid
# means the attachment fouls the clip. Exported as ASCII on purpose --
# an empty binary STL still writes an 84-byte header, so file size
# alone would not tell us anything.
check: | $(OUT)
	@fail=0; \
	run() { rm -f $(OUT)/chk_$$1.stl; \
	  $(SCAD) -o $(OUT)/chk_$$1.stl -D 'part="none"' -D "check=\"$$1\"" \
	    $(CUPD) checks.scad >/dev/null 2>&1; }; \
	solid() { grep -q vertex $(OUT)/chk_$$1.stl 2>/dev/null; }; \
	for c in exists_caddy exists_cup exists_fused; do run $$c; \
	  if solid $$c; then echo "ok    $$c (renders)"; \
	  else echo "FAIL  $$c: part is EMPTY -- every clearance check below is meaningless"; fail=1; fi; \
	done; \
	for c in arc_bar arm_bar mount cup_bars liftoff fused_bars fused_lift; do run $$c; \
	  if solid $$c; then echo "FAIL  $$c: interference"; fail=1; \
	  else echo "ok    $$c (clear)"; fi; \
	done; \
	for c in trapped; do run $$c; \
	  if solid $$c; then echo "ok    $$c (blocked, as required)"; \
	  else echo "FAIL  $$c: the hook does not block release"; fail=1; fi; \
	done; \
	exit $$fail

preview: | $(OUT)
	$(XVFB) $(SCAD) -o $(OUT)/assembly.png --imgsize=1000,900 \
	  --viewall --autocenter --camera=0,0,0,68,0,28,0 \
	  -D 'part="assembly"' stroller_clip.scad
	$(XVFB) $(SCAD) -o $(OUT)/front.png --imgsize=900,980 --projection=o \
	  --viewall --autocenter --camera=0,0,0,0,0,0,0 \
	  -D 'part="assembly"' stroller_clip.scad
	$(XVFB) $(SCAD) -o $(OUT)/raised.png --imgsize=900,980 --projection=o \
	  --viewall --autocenter --camera=0,0,0,0,0,0,0 \
	  -D 'part="raised"' stroller_clip.scad

clean:
	rm -rf $(OUT)
