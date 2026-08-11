SCAD  ?= openscad
OUT   ?= build
PARTS  = hook claw cupholder testfit

# xvfb-run is only needed for the PNG previews; STL export is headless.
XVFB  ?= $(shell command -v xvfb-run 2>/dev/null) $(if $(shell command -v xvfb-run 2>/dev/null),-a,)

.PHONY: all check preview clean

all: $(PARTS:%=$(OUT)/%.stl) $(OUT)/gauge.stl

$(OUT):
	mkdir -p $(OUT)

$(OUT)/gauge.stl: gauge.scad | $(OUT)
	$(SCAD) --export-format=binstl -o $@ $<

$(OUT)/%.stl: stroller_clip.scad | $(OUT)
	$(SCAD) --export-format=binstl -o $@ -D 'part="$*"' $<

# Clearance checks. Each must produce an empty result; anything solid
# means the attachment fouls the clip. Exported as ASCII on purpose --
# an empty binary STL still writes an 84-byte header, so file size
# alone would not tell us anything.
check: | $(OUT)
	@fail=0; \
	for c in hook_bar claw_bar pieces mount cup_bars liftoff; do \
	  rm -f $(OUT)/chk_$$c.stl; \
	  $(SCAD) -o $(OUT)/chk_$$c.stl -D 'part="none"' -D "check=\"$$c\"" \
	    checks.scad >/dev/null 2>&1; \
	  if grep -q vertex $(OUT)/chk_$$c.stl 2>/dev/null; then \
	    echo "FAIL  $$c: interference"; fail=1; \
	  else echo "ok    $$c"; fi; \
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
