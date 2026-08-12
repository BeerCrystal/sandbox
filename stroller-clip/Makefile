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
#
# The exit code is checked too, and it matters more than it looks.
# openscad exits 0 for a solid result and 1 for an empty one, so ANY
# other code -- a crash, an out-of-memory, a timeout kill -- also leaves
# no file, and "no file" is exactly what a passing clearance check looks
# like. A killed render would report itself as clear. That is not
# hypothetical: it happened during development on the liftoff sweep,
# which is slow enough to hit a timeout, and reported green while
# genuinely interfering.
check: | $(OUT)
	@fail=0; \
	run() { rm -f $(OUT)/chk_$$1.stl; \
	  $(SCAD) -o $(OUT)/chk_$$1.stl -D 'part="none"' -D "check=\"$$1\"" \
	    $(CUPD) checks.scad >/dev/null 2>&1; \
	  rc=$$?; \
	  if [ $$rc -ne 0 ] && [ $$rc -ne 1 ]; then \
	    echo "FAIL  $$1: openscad exited $$rc -- crashed or killed, NOT clear"; \
	    return 9; \
	  fi; return 0; }; \
	solid() { grep -q vertex $(OUT)/chk_$$1.stl 2>/dev/null; }; \
	for c in exists_caddy exists_cup exists_fused; do run $$c; \
	  if solid $$c; then echo "ok    $$c (renders)"; \
	  else echo "FAIL  $$c: part is EMPTY -- every clearance check below is meaningless"; fail=1; fi; \
	done; \
	for c in arc_bar arm_bar mount cup_bars insert fused_insert liftoff fused_bars fused_lift; do \
	  if ! run $$c; then fail=1; continue; fi; \
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
	@# Looking down the handle. This is the one that shows the cup's bore
	@# standing clear of both bars -- the front view stacks them in
	@# projection and makes a clear gap look like an overlap.
	$(XVFB) $(SCAD) -o $(OUT)/plan.png --imgsize=900,700 --projection=o \
	  --viewall --autocenter --camera=0,0,0,90,0,0,0 \
	  -D 'part="assembly"' stroller_clip.scad

clean:
	rm -rf $(OUT)
