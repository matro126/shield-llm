#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from check_annotation_diff_views import classify, view_of  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
STATE_NAME = "fix_views_state.json"

THUMB_MAX = 300
BG = "#15171c"
FG = "#e6e8ec"
MUTED = "#8b91a1"
ACCENT_F = "#4f9dff"
ACCENT_L = "#ffb347"
ACCENT_SEL = "#4ade80"
ACCENT_CUR = "#c084fc"
CURRENT_BG = "#2a2f3d"
WARN = "#f87171"


def folder_images(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def select_samples(annotation: dict, include_ok: bool) -> list[tuple[str, str, dict]]:
    out = []
    for split, records in annotation.items():
        for record in records:
            status, *_ = classify(record["image_path"])
            if include_ok or status != "OK":
                out.append((split, status, record))
    return out


def ordered_pair(names: list[str]) -> list[str]:
    views = [view_of(n) for n in names]
    if sorted(v for v in views if v) == ["frontal", "lateral"]:
        return sorted(names, key=lambda n: 0 if view_of(n) == "frontal" else 1)
    return list(names)


def load_state(out: Path) -> dict:
    path = out / STATE_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"decided": {}, "skipped": []}


def save_state(out: Path, state: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / STATE_NAME).write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class Picker:
    def __init__(self, todo, images_dir: Path, out: Path, state: dict):
        self.todo = todo
        self.images_dir = images_dir
        self.out = out
        self.state = state
        self.index = 0
        self.selection: list[str] = []
        self.names: list[str] = []
        self.thumbs: list = []
        self.boxes: list = []

        self.root = tk.Tk()
        self.root.title("Correzione viste in annotation.json")
        self.root.configure(bg=BG)

        self.header = tk.Label(self.root, text="", bg=BG, fg=FG,
                               font=("Helvetica", 15, "bold"), anchor="w")
        self.header.pack(fill="x", padx=16, pady=(14, 0))
        self.sub = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                            font=("Helvetica", 11), anchor="w", justify="left")
        self.sub.pack(fill="x", padx=16, pady=(2, 10))

        self.strip = tk.Frame(self.root, bg=BG)
        self.strip.pack(padx=16)

        self.status = tk.Label(self.root, text="", bg=BG, fg=MUTED,
                               font=("Helvetica", 12))
        self.status.pack(pady=(12, 4))
        tk.Label(self.root,
                 text="1..9 seleziona · Invio conferma · Backspace azzera · "
                      "S salta · Esc esci",
                 bg=BG, fg=MUTED, font=("Helvetica", 10)).pack(pady=(0, 12))

        for digit in range(1, 10):
            self.root.bind(str(digit), lambda e, d=digit: self.toggle_index(d - 1))
        self.root.bind("<Return>", lambda e: self.confirm())
        self.root.bind("<BackSpace>", lambda e: self.clear())
        for key in ("s", "S"):
            self.root.bind(key, lambda e: self.skip())
        for key in ("<Escape>", "q", "Q"):
            self.root.bind(key, lambda e: self.quit())

        self.show()

    def _thumbnail(self, path: Path):
        try:
            from PIL import Image, ImageTk
            image = Image.open(path)
            image.thumbnail((THUMB_MAX, THUMB_MAX))
            return ImageTk.PhotoImage(image)
        except Exception:
            photo = tk.PhotoImage(file=str(path))
            factor = max(1, max(photo.width(), photo.height()) // THUMB_MAX)
            return photo.subsample(factor, factor) if factor > 1 else photo

    def show(self):
        if self.index >= len(self.todo):
            return self.quit()

        split, status, record = self.todo[self.index]
        sample_id = str(record["id"])
        folder = self.images_dir / sample_id
        self.names = folder_images(folder) if folder.is_dir() else []
        self.selection = list(self.state["decided"].get(sample_id, []))

        referenced = {Path(p).name for p in record["image_path"]}
        self.header.config(
            text=f"[{self.index + 1}/{len(self.todo)}]  {sample_id}   ({split})"
        )
        self.sub.config(
            text=f"stato: {status}    ora in annotation: "
                 f"{', '.join(sorted(referenced)) or '—'}\n"
                 f"nella cartella: {len(self.names)} immagini — sfondo chiaro e "
                 f"'● ORA IN ANNOTATION' = quelle che stai sostituendo"
        )

        for box in self.boxes:
            box.destroy()
        self.boxes, self.thumbs = [], []

        for position, name in enumerate(self.names):
            current = name in referenced
            panel = CURRENT_BG if current else BG
            box = tk.Frame(self.strip, bg=panel, highlightthickness=3,
                           highlightbackground=BG)
            box.grid(row=position // 5, column=position % 5, padx=6, pady=6)
            thumb = self._thumbnail(folder / name)
            self.thumbs.append(thumb)
            tk.Label(box, image=thumb, bg=panel).pack(padx=4, pady=(4, 0))
            view = view_of(name)
            colour = {"frontal": ACCENT_F, "lateral": ACCENT_L}.get(view, WARN)
            tk.Label(box, text=f"{position + 1}. {name}", bg=panel, fg=colour,
                     font=("Helvetica", 10, "bold")).pack()
            tk.Label(box,
                     text="● ORA IN ANNOTATION" if current else "extra (non referenziata)",
                     bg=panel, fg=ACCENT_CUR if current else MUTED,
                     font=("Helvetica", 9, "bold" if current else "normal")).pack(
                         pady=(0, 4))
            box.bind("<Button-1>", lambda e, p=position: self.toggle_index(p))
            for child in box.winfo_children():
                child.bind("<Button-1>", lambda e, p=position: self.toggle_index(p))
            self.boxes.append(box)

        self.refresh()

    def refresh(self):
        for position, name in enumerate(self.names):
            chosen = name in self.selection
            self.boxes[position].config(
                highlightbackground=ACCENT_SEL if chosen else BG
            )
        views = sorted(v for v in (view_of(n) for n in self.selection) if v)
        if len(self.selection) != 2:
            text, colour = f"selezionate {len(self.selection)}/2", MUTED
        elif views == ["frontal", "lateral"]:
            text, colour = "1 frontale + 1 laterale — Invio per confermare", ACCENT_SEL
        else:
            text, colour = ("attenzione: non e' 1 frontale + 1 laterale — "
                            "Invio conferma comunque"), WARN
        self.status.config(text=text, fg=colour)

    def toggle_index(self, position: int):
        if position >= len(self.names):
            return
        name = self.names[position]
        if name in self.selection:
            self.selection.remove(name)
        elif len(self.selection) < 2:
            self.selection.append(name)
        self.refresh()

    def clear(self):
        self.selection = []
        self.refresh()

    def confirm(self):
        if len(self.selection) != 2:
            return
        sample_id = str(self.todo[self.index][2]["id"])
        self.state["decided"][sample_id] = ordered_pair(self.selection)
        if sample_id in self.state["skipped"]:
            self.state["skipped"].remove(sample_id)
        save_state(self.out, self.state)
        self.index += 1
        self.show()

    def skip(self):
        sample_id = str(self.todo[self.index][2]["id"])
        if sample_id not in self.state["skipped"]:
            self.state["skipped"].append(sample_id)
        save_state(self.out, self.state)
        self.index += 1
        self.show()

    def quit(self):
        save_state(self.out, self.state)
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def write_dataset(dataset: Path, out: Path, annotation: dict, decided: dict,
                  images_mode: str) -> None:
    fixed = {}
    for split, records in annotation.items():
        fixed[split] = []
        for record in records:
            sample_id = str(record["id"])
            if sample_id in decided:
                record = {**record,
                          "image_path": [f"{sample_id}/{n}" for n in decided[sample_id]]}
            fixed[split].append(record)

    out.mkdir(parents=True, exist_ok=True)
    (out / "annotation.json").write_text(
        json.dumps(fixed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if images_mode != "none":
        images_out = out / "images"
        if images_out.exists():
            shutil.rmtree(images_out)
        images_out.mkdir()
        for records in fixed.values():
            for record in records:
                sample_id = str(record["id"])
                source = dataset / "images" / sample_id
                destination = images_out / sample_id
                if images_mode == "copy":
                    shutil.copytree(source, destination)
                else:
                    destination.symlink_to(source.resolve(), target_is_directory=True)

    for extra in ("labeling_state.json", "removed_no_impression.txt",
                  "removed_same_view_pairs.txt"):
        if (dataset / extra).is_file():
            shutil.copy2(dataset / extra, out / extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset di input (annotation.json + images/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="dove scrivere il dataset corretto")
    ap.add_argument("--images", choices=("copy", "symlink", "none"), default="copy",
                    help="come portare le immagini nel dataset corretto (default: copy)")
    ap.add_argument("--include-ok", action="store_true",
                    help="passa in rassegna tutti i sample, non solo i non conformi")
    ap.add_argument("--list", action="store_true",
                    help="elenca i sample da correggere ed esci, senza aprire la GUI")
    ap.add_argument("--redo", action="store_true",
                    help="ignora le decisioni gia' prese e ricomincia")
    args = ap.parse_args()

    ann_path = args.dataset / "annotation.json"
    images_dir = args.dataset / "images"
    for path in (ann_path, images_dir):
        if not path.exists():
            sys.exit(f"Percorso non trovato: {path}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    samples = select_samples(annotation, args.include_ok)

    if not samples:
        print("Nessun sample da correggere: tutti hanno 1 frontale + 1 laterale.")
        return 0

    state = {"decided": {}, "skipped": []} if args.redo else load_state(args.out)

    print(f"dataset: {args.dataset}")
    print(f"da rivedere: {len(samples)} sample "
          f"(gia' decisi {len(state['decided'])}, saltati {len(state['skipped'])})\n")
    for split, status, record in samples:
        sample_id = str(record["id"])
        folder = images_dir / sample_id
        n_images = len(folder_images(folder)) if folder.is_dir() else 0
        mark = ("✓" if sample_id in state["decided"]
                else "–" if sample_id in state["skipped"] else " ")
        print(f"  {mark} {sample_id:<26}{split:<7}{status:<16}"
              f"{n_images} immagini in cartella")

    if args.list:
        return 0

    todo = [s for s in samples if str(s[2]["id"]) not in state["decided"]]
    if todo:
        print(f"\nApro la GUI su {len(todo)} sample…")
        Picker(todo, images_dir, args.out, state).run()
    else:
        print("\nTutti i sample hanno gia' una decisione: scrivo il dataset.")

    write_dataset(args.dataset, args.out, annotation, state["decided"], args.images)
    print(f"\ncorretti: {len(state['decided'])}   saltati: {len(state['skipped'])}")
    print(f"dataset:  {args.out}")
    print(f"stato:    {args.out / STATE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
