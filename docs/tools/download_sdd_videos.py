"""Download single files of the Stanford Drone Dataset without fetching the whole 71 GB zip.

    uv run --python 3.12 --with libtorrent python docs/tools/download_sdd_videos.py OUTPUT_DIR \
        videos/gates/video4/video.mov annotations/gates/video4/annotations.txt ...

The dataset is distributed as one zip file (official server, or Academic Torrents). A zip file has its table of
contents at the end, so this script downloads the last torrent pieces first, reads the table of contents, and then
downloads only the pieces that contain the requested files. Without file arguments it lists the contents.

The data is licensed CC BY-NC-SA 3.0 (non-commercial) by its authors; see https://cvgl.stanford.edu/projects/uav_data/.
"""

from __future__ import annotations

import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import libtorrent as lt

TORRENT_URL = "https://academictorrents.com/download/01f95ea32e160e6c251ea55a87bd5a24b23cb03d.torrent"
DEFAULT_FILES = [f"{kind}/gates/video{v}/{name}" for v in (4, 5, 7)
                 for kind, name in (("videos", "video.mov"), ("annotations", "annotations.txt"))]


class SparseZip:
    def __init__(self, work: Path):
        work.mkdir(parents=True, exist_ok=True)
        torrent = work / "sdd.torrent"
        if not torrent.is_file():
            request = urllib.request.Request(TORRENT_URL, headers={"User-Agent": "Mozilla/5.0 (cyclist-tracker)"})
            torrent.write_bytes(urllib.request.urlopen(request, timeout=60).read())
        self.info = lt.torrent_info(str(torrent))
        self.size = self.info.total_size()
        self.session = lt.session({"listen_interfaces": "0.0.0.0:6881", "enable_dht": True})
        params = lt.add_torrent_params()
        params.ti, params.save_path = self.info, str(work)
        params.storage_mode = lt.storage_mode_t.storage_mode_sparse
        self.handle = self.session.add_torrent(params)
        self.handle.prioritize_pieces([0] * self.info.num_pieces())
        self.path = work / self.info.files().file_path(0)

    def fetch(self, start: int, end: int) -> None:
        """Download the pieces that contain bytes [start, end)."""
        length = self.info.piece_length()
        pieces = range(start // length, (min(end, self.size) - 1) // length + 1)
        for i in pieces:
            self.handle.piece_priority(i, 7)
        while not all(self.handle.have_piece(i) for i in pieces):
            s = self.handle.status()
            done = sum(self.handle.have_piece(i) for i in pieces)
            print(f"\r  {done}/{len(pieces)} pieces, {s.num_peers} peers, {s.download_rate / 1e6:.1f} MB/s   ",
                  end="", flush=True)
            time.sleep(2)
        time.sleep(1)  # let libtorrent flush the data to disk
        print()

    def read(self, offset: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def entries(self) -> dict[str, tuple[int, int, int]]:
        """name -> (local header offset, compressed size, method), from the zip's central directory."""
        self.fetch(self.size - 70000, self.size)
        tail = self.read(self.size - 70000, 70000)
        end = tail.rfind(b"PK\x05\x06")
        cd_size, cd_offset = struct.unpack("<II", tail[end + 12:end + 20])
        if cd_offset == 0xFFFFFFFF:  # zip64
            locator = tail.rfind(b"PK\x06\x07")
            record = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
            self.fetch(record, record + 56)
            cd_size, cd_offset = struct.unpack("<QQ", self.read(record, 56)[40:56])
        self.fetch(cd_offset, cd_offset + cd_size)
        cd, i, result = self.read(cd_offset, cd_size), 0, {}
        while cd[i:i + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[i + 10:i + 12])
            csize, usize = struct.unpack("<II", cd[i + 20:i + 28])
            nl, xl, cl = struct.unpack("<HHH", cd[i + 28:i + 34])
            offset, = struct.unpack("<I", cd[i + 42:i + 46])
            name = cd[i + 46:i + 46 + nl].decode()
            extra, j = cd[i + 46 + nl:i + 46 + nl + xl], 0
            while j < len(extra):
                tag, size = struct.unpack("<HH", extra[j:j + 4])
                if tag == 1:  # zip64 sizes and offset
                    values = list(struct.unpack("<" + "Q" * (size // 8), extra[j + 4:j + 4 + size]))
                    if usize == 0xFFFFFFFF:
                        usize = values.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = values.pop(0)
                    if offset == 0xFFFFFFFF:
                        offset = values.pop(0)
                j += 4 + size
            result[name] = (offset, csize, method)
            i += 46 + nl + xl + cl
        return result

    def extract(self, entry: tuple[int, int, int], target: Path) -> None:
        offset, csize, method = entry
        self.fetch(offset, offset + csize + 1024)
        header = self.read(offset, 30)
        nl, xl = struct.unpack("<HH", header[26:30])
        target.parent.mkdir(parents=True, exist_ok=True)
        data_start, left = offset + 30 + nl + xl, csize
        inflate = zlib.decompressobj(-15) if method == 8 else None
        with open(self.path, "rb") as src, open(target, "wb") as out:
            src.seek(data_start)
            while left:
                chunk = src.read(min(left, 1 << 24))
                left -= len(chunk)
                out.write(inflate.decompress(chunk) if inflate else chunk)
            if inflate:
                out.write(inflate.flush())


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    output = Path(sys.argv[1])
    wanted = sys.argv[2:] or DEFAULT_FILES
    archive = SparseZip(output / ".download")
    print("Reading the table of contents …")
    entries = {name.removeprefix("stanford_campus_dataset/"): e for name, e in archive.entries().items()}
    if sys.argv[2:] == ["--list"]:
        print("\n".join(sorted(entries)))
        return
    for name in wanted:
        if name not in entries:
            sys.exit(f"Not in the dataset: {name} (use --list)")
        print(f"Downloading {name} ({entries[name][1] / 1e6:.0f} MB compressed)")
        archive.extract(entries[name], output / name)
    print(f"Done. Extracted files are in {output}; the sparse download in {output / '.download'} can be deleted.")


if __name__ == "__main__":
    main()
