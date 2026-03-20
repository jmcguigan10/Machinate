import fs from "node:fs";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const [, , url, outputPath] = process.argv;

if (!url || !outputPath) {
  console.error("usage: node fetch_url.mjs <url> <output-path>");
  process.exit(1);
}

await mkdir(path.dirname(outputPath), { recursive: true });

const response = await fetch(url, {
  redirect: "follow",
  headers: {
    "user-agent": "machinator-fetch/0.1",
  },
});

if (!response.ok || !response.body) {
  console.error(`download failed: ${response.status} ${response.statusText}`);
  process.exit(1);
}

await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(outputPath));

const sizeBytes = fs.statSync(outputPath).size;
console.log(JSON.stringify({
  url,
  final_url: response.url,
  output_path: outputPath,
  content_type: response.headers.get("content-type") || "",
  size_bytes: sizeBytes,
}));
