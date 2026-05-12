// GliaAnalysis — headless threshold-only macro
//
// Thresholds every TIFF in input_dir using the same chain as
// MicrogliaMorphology_Program.ijm (Enhance Contrast 0.35 → Unsharp Mask r=3
// m=0.6 → Despeckle → Auto/Local Threshold → Despeckle → Close- → Remove
// Outliers r=2 t=50). Writes results to output_dir/.
//
// Required args (pipe-delimited key=value via getArgument()):
//   input_dir, output_dir, threshold_kind, threshold_method,
//   manual_lower, manual_upper, local_radius, preprocess
//
// Outputs:
//   output_dir/<name>_thresholded.tif
//   output_dir/Areas.csv
//
// Single-cell extraction (Analyze Particles) is intentionally NOT here —
// FIJI's RoiManager fails with HeadlessException in --headless mode, so
// per-cell cropping happens in Python (glia.segment).

arg_string = getArgument();
if (lengthOf(arg_string) == 0) exit("threshold.ijm: no arguments.");

pairs = split(arg_string, "|");
input_dir = "";  output_dir = "";
threshold_kind = "global";  threshold_method = "Otsu";
manual_lower = 0;  manual_upper = 255;
local_radius = 100;
preprocess = 1;

for (k = 0; k < pairs.length; k++) {
    kv = split(pairs[k], "=");
    if (kv.length != 2) continue;
    if      (kv[0] == "input_dir")        input_dir = kv[1];
    else if (kv[0] == "output_dir")       output_dir = kv[1];
    else if (kv[0] == "threshold_kind")   threshold_kind = kv[1];
    else if (kv[0] == "threshold_method") threshold_method = kv[1];
    else if (kv[0] == "manual_lower")     manual_lower = parseInt(kv[1]);
    else if (kv[0] == "manual_upper")     manual_upper = parseInt(kv[1]);
    else if (kv[0] == "local_radius")     local_radius = parseInt(kv[1]);
    else if (kv[0] == "preprocess")       preprocess = parseInt(kv[1]);
}
if (input_dir == "" || output_dir == "") exit("threshold.ijm: input_dir and output_dir required.");
if (!endsWith(input_dir, File.separator))   input_dir  = input_dir  + File.separator;
if (!endsWith(output_dir, File.separator))  output_dir = output_dir + File.separator;
File.makeDirectory(output_dir);

print("=== threshold.ijm ===");
print("  input_dir       = " + input_dir);
print("  output_dir      = " + output_dir);
print("  threshold_kind  = " + threshold_kind);
print("  threshold_method= " + threshold_method);

setBatchMode(true);
inputs = Array.sort(getFileList(input_dir));

for (i = 0; i < inputs.length; i++) {
    fname = inputs[i];
    if (!endsWith(toLowerCase(fname), ".tif") && !endsWith(toLowerCase(fname), ".tiff")) continue;
    print("[" + (i + 1) + "/" + inputs.length + "] " + fname);

    open(input_dir + fname);
    run("Set Measurements...", "area display redirect=None decimal=9");
    run("Measure");

    run("8-bit");
    run("Grays");
    if (preprocess == 1) {
        run("Enhance Contrast", "saturated=0.35");
        run("Unsharp Mask...", "radius=3 mask=0.60");
        run("Despeckle");
    }
    if (threshold_kind == "manual") {
        setThreshold(manual_lower, manual_upper);
        setOption("BlackBackground", true);
        run("Convert to Mask");
    } else if (threshold_kind == "local") {
        run("Auto Local Threshold",
            "method=" + threshold_method
          + " radius=" + local_radius
          + " parameter_1=0 parameter_2=0 white");
    } else {
        run("Auto Threshold",
            "method=" + threshold_method + " ignore_black white");
    }
    run("Despeckle");
    run("Close-");
    run("Remove Outliers...", "radius=2 threshold=50 which=Bright");

    saveAs("Tiff", output_dir + fname + "_thresholded");
    close("*");
}

saveAs("Results", output_dir + "Areas.csv");
close("Results");
setBatchMode(false);
print("=== threshold.ijm done ===");
// Force the JVM to exit. Without this, --headless leaves background threads
// alive and subprocess.run hits its timeout.
eval("script", "System.exit(0);");
