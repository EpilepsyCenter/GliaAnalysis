// GliaAnalysis — Olympus VSI → OME-TIFF conversion macro
//
// Bio-Formats is the only path to read .vsi reliably; there is no native
// Python reader. This macro takes a directory of .vsi files and writes the
// full-resolution image of each to output_dir/<stem>.ome.tif. The rest of
// the GliaAnalysis pipeline then treats those OME-TIFFs as ordinary
// multi-channel inputs (channel pick + Z-projection in glia/io.py).
//
// VSI files contain a pyramid: series 0 is usually the macro/label image,
// series 1+ are the actual data. We inspect every series with the
// Bio-Formats Macro Extensions and pick the one with the largest XY
// footprint — that's the full-resolution layer regardless of slide layout.
//
// Required args (pipe-delimited key=value via getArgument()):
//   input_dir   — folder containing .vsi files
//   output_dir  — folder for <stem>.ome.tif (created if missing)

arg_string = getArgument();
if (lengthOf(arg_string) == 0) exit("vsi_convert.ijm: no arguments.");

pairs = split(arg_string, "|");
input_dir = "";  output_dir = "";
for (k = 0; k < pairs.length; k++) {
    kv = split(pairs[k], "=");
    if (kv.length != 2) continue;
    if      (kv[0] == "input_dir")  input_dir = kv[1];
    else if (kv[0] == "output_dir") output_dir = kv[1];
}
if (input_dir == "" || output_dir == "") exit("vsi_convert.ijm: input_dir and output_dir required.");
if (!endsWith(input_dir, File.separator))  input_dir  = input_dir  + File.separator;
if (!endsWith(output_dir, File.separator)) output_dir = output_dir + File.separator;
File.makeDirectory(output_dir);

print("=== vsi_convert.ijm ===");
print("  input_dir  = " + input_dir);
print("  output_dir = " + output_dir);

run("Bio-Formats Macro Extensions");
setBatchMode(true);
files = Array.sort(getFileList(input_dir));

for (i = 0; i < files.length; i++) {
    fname = files[i];
    if (!endsWith(toLowerCase(fname), ".vsi")) continue;

    src = input_dir + fname;
    stem = substring(fname, 0, lengthOf(fname) - 4);
    dst = output_dir + stem + ".ome.tif";

    if (File.exists(dst)) {
        // Skip if the converted file is newer than the source.
        if (File.dateLastModified(dst) >= File.dateLastModified(src)) {
            print("[" + (i + 1) + "/" + files.length + "] cached: " + fname);
            continue;
        }
    }
    print("[" + (i + 1) + "/" + files.length + "] converting: " + fname);

    // Find the series with the largest XY (= full resolution).
    Ext.setId(src);
    Ext.getSeriesCount(nSeries);
    bestSeries = 0;
    bestPixels = -1;
    for (s = 0; s < nSeries; s++) {
        Ext.setSeries(s);
        Ext.getSizeX(sx);
        Ext.getSizeY(sy);
        px = sx * sy;
        if (px > bestPixels) {
            bestPixels = px;
            bestSeries = s;
        }
    }
    Ext.close();
    print("    chose series " + bestSeries + " (" + bestPixels + " px)");

    // Bio-Formats series flags are 1-indexed in the importer args.
    seriesArg = "series_" + (bestSeries + 1) + "=true";
    run("Bio-Formats Importer",
        "open=[" + src + "]"
      + " color_mode=Default view=Hyperstack stack_order=XYCZT"
      + " " + seriesArg);

    saveAs("OME-TIFF", dst);
    close("*");
}

setBatchMode(false);
print("=== vsi_convert.ijm done ===");
eval("script", "System.exit(0);");
