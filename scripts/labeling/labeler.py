#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tkinter as tk

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "iu_xray_r2gen_labeled")
IMAGES = os.path.join(WORK, "images")
ANNOTATION = os.path.join(WORK, "annotation.json")
STATE = os.path.join(WORK, "labeling_state.json")

LABELS = {"frontal": "frontal", "lateral": "lateral"}
MAX_IMG_H = 700          
THUMB_SUBSAMPLE = 3

BG = "#15171c"
FG = "#e6e8ec"
MUTED = "#8b91a1"
ACCENT_F = "#4f9dff"
ACCENT_L = "#ffb347"

def natural_key(name):
    stem = os.path.splitext(name)[0]
    return (0, int(stem), "") if stem.isdigit() else (1, 0, name)


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def new_names_for(decisions):
    counts = {}
    out = []
    for orig, label in decisions:
        counts[label] = counts.get(label, 0) + 1
        n = counts[label]
        new = f"{label}.png" if n == 1 else f"{label}_{n}.png"
        out.append((orig, new, label))
    return out


class Dataset:
    def __init__(self):
        if not os.path.isdir(WORK):
            sys.exit(f"Copia di lavoro non trovata: {WORK}\n"
                     f"Lancia prima:  python3 prepare_copy.py")

        self.annotation = json.load(open(ANNOTATION))
        self.by_id = {}
        for split in self.annotation.values():
            for rec in split:
                self.by_id[rec["id"]] = rec

        on_disk = {d for d in os.listdir(IMAGES)
                   if os.path.isdir(os.path.join(IMAGES, d))}
        self.studies = [sid for sid in self.by_id if sid in on_disk]
        self.studies += sorted(on_disk - set(self.by_id))

        self.state = self._load_state()
        self._recover()


    def _load_state(self):
        if os.path.exists(STATE):
            with open(STATE) as fh:
                return json.load(fh)
        return {"version": 1, "done": {}, "skipped": [], "pending": None}

    def save_state(self):
        atomic_write_json(STATE, self.state)

    def _recover(self):
        pending = self.state.get("pending")
        if not pending:
            return
        print(f"[recover] commit interrotto su {pending['study']}: lo riapplico")
        self._apply(pending["study"], pending["entries"])
        self.state["done"][pending["study"]] = pending["entries"]
        self.state["pending"] = None
        self.save_state()


    def _apply(self, study, entries):
        folder = os.path.join(IMAGES, study)
        for orig, new, _label in entries:
            if orig == new:
                continue
            src = os.path.join(folder, orig)
            dst = os.path.join(folder, new)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

        rename_map = {orig: new for orig, new, _ in entries}
        rec = self.by_id.get(study)
        if rec is not None:
            changed = False
            paths = []
            for p in rec["image_path"]:
                d, _, name = p.rpartition("/")
                if name in rename_map and rename_map[name] != name:
                    paths.append(f"{d}/{rename_map[name]}")
                    changed = True
                else:
                    paths.append(p)
            if changed:
                rec["image_path"] = paths
                atomic_write_json(ANNOTATION, self.annotation)

    def commit(self, study, decisions):
        entries = new_names_for(decisions)
        self.state["pending"] = {"study": study, "entries": entries}
        self.save_state()                      
        self._apply(study, entries)            
        self.state["done"][study] = entries    
        self.state["pending"] = None
        if study in self.state["skipped"]:
            self.state["skipped"].remove(study)
        self.save_state()

    def uncommit(self, study):
        entries = self.state["done"].get(study)
        if not entries:
            return None
        reverse = [(new, orig, label) for orig, new, label in entries]
        self.state["pending"] = {"study": study, "entries": reverse}
        self.save_state()
        self._apply(study, reverse)
        self.state["done"].pop(study, None)
        self.state["pending"] = None
        self.save_state()
        return entries

    def skip(self, study):
        if study not in self.state["skipped"]:
            self.state["skipped"].append(study)
            self.save_state()

    def images_of(self, study):
        folder = os.path.join(IMAGES, study)
        return sorted((f for f in os.listdir(folder)
                       if f.lower().endswith(".png")), key=natural_key)

    def todo(self, include_skipped):
        done = self.state["done"]
        skipped = set() if include_skipped else set(self.state["skipped"])
        return [s for s in self.studies if s not in done and s not in skipped]

    def verify(self):
        problems = []
        for study, entries in self.state["done"].items():
            folder = os.path.join(IMAGES, study)
            for orig, new, _ in entries:
                if not os.path.exists(os.path.join(folder, new)):
                    problems.append(f"{study}: manca il file {new} (era {orig})")
            rec = self.by_id.get(study)
            if rec is None:
                continue
            for p in rec["image_path"]:
                if not os.path.exists(os.path.join(IMAGES, p)):
                    problems.append(f"{study}: annotation punta a {p} che non esiste")
        for split in self.annotation.values():
            for rec in split:
                for p in rec["image_path"]:
                    if not os.path.exists(os.path.join(IMAGES, p)):
                        problems.append(f"annotation: percorso rotto {p}")
        return sorted(set(problems))

class App:
    def __init__(self, ds, queue):
        self.ds = ds
        self.queue = queue
        self.qi = 0                 
        self.decisions = []         
        self.images = []            
        self._refs = []             

        self.root = tk.Tk()
        self.root.title("Etichettatura frontale / laterale")
        self.root.configure(bg=BG)

        self.header = tk.Label(self.root, bg=BG, fg=FG,
                               font=("Helvetica", 15, "bold"), anchor="w")
        self.header.pack(fill="x", padx=16, pady=(14, 0))

        self.sub = tk.Label(self.root, bg=BG, fg=MUTED,
                            font=("Helvetica", 12), anchor="w")
        self.sub.pack(fill="x", padx=16, pady=(2, 10))

        body = tk.Frame(self.root, bg=BG)
        body.pack(padx=16)

        self.canvas = tk.Canvas(body, width=512, height=MAX_IMG_H,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(side="left")

        self.side = tk.Frame(body, bg=BG, width=210)
        self.side.pack(side="left", fill="y", padx=(16, 0))

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(pady=14)
        self._button(bar, "F  —  Frontale", ACCENT_F, lambda: self.label("frontal"))
        self._button(bar, "L  —  Laterale", ACCENT_L, lambda: self.label("lateral"))
        self._button(bar, "Backspace  —  Indietro", "#3a3f4b", self.back)
        self._button(bar, "S  —  Salta studio", "#3a3f4b", self.skip)

        self.status = tk.Label(self.root, bg=BG, fg=MUTED,
                               font=("Helvetica", 11))
        self.status.pack(pady=(0, 12))

        for k in ("f", "F", "<Left>"):
            self.root.bind(k, lambda e: self.label("frontal"))
        for k in ("l", "L", "<Right>"):
            self.root.bind(k, lambda e: self.label("lateral"))
        self.root.bind("<BackSpace>", lambda e: self.back())
        for k in ("s", "S"):
            self.root.bind(k, lambda e: self.skip())
        for k in ("q", "Q", "<Escape>"):
            self.root.bind(k, lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.load_study()

    def _button(self, parent, text, color, cmd):
        b = tk.Label(parent, text=text, bg=color, fg="#0d0f13" if color != "#3a3f4b" else FG,
                     font=("Helvetica", 12, "bold"), padx=16, pady=9, cursor="hand2")
        b.pack(side="left", padx=5)
        b.bind("<Button-1>", lambda e: cmd())
        return b


    def load_study(self):
        if self.qi >= len(self.queue):
            return self.finish()
        study = self.queue[self.qi]
        self.images = self.ds.images_of(study)
        self.decisions = []
        self.render()

    def render(self):
        study = self.queue[self.qi]
        i = len(self.decisions)
        total_done = len(self.ds.state["done"])

        self.header.config(text=f"{study}      —      immagine {i + 1} di {len(self.images)}")
        self.sub.config(text=f"studio {self.qi + 1}/{len(self.queue)} di questa sessione   ·   "
                             f"{total_done}/{len(self.ds.studies)} studi completati nel dataset   ·   "
                             f"{len(self.ds.state['skipped'])} saltati")

        self._refs.clear()
        self.canvas.delete("all")
        img = self._photo(os.path.join(IMAGES, study, self.images[i]))
        self.canvas.create_image(256, MAX_IMG_H // 2, image=img)
        self._refs.append(img)

        for w in self.side.winfo_children():
            w.destroy()
        tk.Label(self.side, text="tutte le immagini\ndello studio", bg=BG, fg=MUTED,
                 font=("Helvetica", 10), justify="left").pack(anchor="w", pady=(0, 8))
        assigned = dict(self.decisions)
        for j, name in enumerate(self.images):
            box = tk.Frame(self.side, bg=ACCENT_F if j == i else BG, padx=2, pady=2)
            box.pack(anchor="w", pady=4)
            th = self._photo(os.path.join(IMAGES, study, name), thumb=True)
            tk.Label(box, image=th, bg=BG).pack()
            self._refs.append(th)
            lab = assigned.get(name)
            txt = f"{name} → {lab}" if lab else (f"{name}  ←" if j == i else name)
            tk.Label(box, text=txt, bg=BG,
                     fg=(ACCENT_F if lab == "frontal" else ACCENT_L if lab else
                         FG if j == i else MUTED),
                     font=("Helvetica", 10)).pack()

        self.status.config(text="F = frontale   ·   L = laterale   ·   Backspace = indietro   "
                                "·   S = salta   ·   Esc = esci")

    def _photo(self, path, thumb=False):
        img = tk.PhotoImage(file=path)
        if thumb:
            return img.subsample(THUMB_SUBSAMPLE, THUMB_SUBSAMPLE)
        if img.height() > MAX_IMG_H:
            return img.subsample(2, 2)
        return img


    def label(self, label):
        if self.qi >= len(self.queue):
            return
        i = len(self.decisions)
        self.decisions.append((self.images[i], label))
        if len(self.decisions) == len(self.images):
            self.ds.commit(self.queue[self.qi], self.decisions)
            self.qi += 1
            self.load_study()
        else:
            self.render()

    def back(self):
        if self.decisions:                     
            self.decisions.pop()
            self.render()
        elif self.qi > 0:                       
            self.qi -= 1
            prev = self.queue[self.qi]
            self.ds.uncommit(prev)
            self.images = self.ds.images_of(prev)
            self.decisions = []
            self.render()

    def skip(self):
        if self.qi >= len(self.queue):
            return
        self.ds.skip(self.queue[self.qi])
        self.qi += 1
        self.load_study()

    def finish(self):
        self.canvas.delete("all")
        for w in self.side.winfo_children():
            w.destroy()
        self.header.config(text="Fatto: nessuno studio rimasto in coda.")
        self.sub.config(text=f"{len(self.ds.state['done'])}/{len(self.ds.studies)} studi "
                             f"completati   ·   {len(self.ds.state['skipped'])} saltati "
                             f"(riprendili con --include-skipped)")
        self.status.config(text="Premi Esc per uscire.")

    def quit(self):
        self.ds.save_state()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-skipped", action="store_true",
                    help="ripresenta anche gli studi saltati in precedenza")
    ap.add_argument("--verify", action="store_true",
                    help="controlla la consistenza fra stato, file e annotation.json ed esce")
    args = ap.parse_args()

    ds = Dataset()

    if args.verify:
        problems = ds.verify()
        done = len(ds.state["done"])
        print(f"studi completati: {done}/{len(ds.studies)}")
        print(f"studi saltati:    {len(ds.state['skipped'])}")
        if problems:
            print(f"\n{len(problems)} problemi:")
            for p in problems[:50]:
                print("  -", p)
            sys.exit(1)
        print("nessun problema: file e annotation.json sono consistenti.")
        return

    queue = ds.todo(args.include_skipped)
    if not queue:
        print("Non ci sono studi da etichettare.")
        if ds.state["skipped"]:
            print(f"({len(ds.state['skipped'])} saltati: rilancia con --include-skipped)")
        return

    print(f"{len(queue)} studi da etichettare "
          f"({len(ds.state['done'])} gia' fatti). Apro la finestra...")
    App(ds, queue).run()


if __name__ == "__main__":
    main()
