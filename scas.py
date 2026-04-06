import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import cv2
import pygame
import numpy as np
from pushbullet import Pushbullet
from ultralytics import YOLO
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# YOLOv26 will automatically download the .pt model file
MODEL_PATH = 'yolo26n.pt'
COCO_NAMES_PATH = os.path.join(SCRIPT_DIR, 'coco.names')
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, 'known_faces')
ALARM_PATH = os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', '86502^alarm.wav')
RECORDINGS_DIR = os.path.join(os.environ.get('USERPROFILE', ''), 'scas_recordings')
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
for d in [RECORDINGS_DIR, KNOWN_FACES_DIR]:
    os.makedirs(d, exist_ok=True)
with open(COCO_NAMES_PATH, 'r') as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]
try:
    yolo_model = YOLO(MODEL_PATH)
    print('[SCAS] YOLOv26 loaded OK')
except Exception as e:
    print(f'[SCAS] Error loading YOLOv26: {e}')
    yolo_model = None
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=150.0)
known_names = []
model_trained = False

def load_known_faces():
    global known_names, model_trained
    faces, labels = ([], [])
    label_map = {}
    
    # 1. Load from known_faces directory (legacy)
    for root, dirs, files in os.walk(KNOWN_FACES_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(fpath)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            name = os.path.basename(root) if os.path.abspath(root) != os.path.abspath(KNOWN_FACES_DIR) else os.path.splitext(fname)[0]
            name = ''.join([i for i in name if not i.isdigit()]).strip().rstrip('_- ')
            
            if name not in label_map: label_map[name] = len(label_map)
            
            # Detect face and augment
            detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(detected) == 0: detected = [(0, 0, gray.shape[1], gray.shape[0])]
            
            for x, y, w, h in detected:
                # Augmentation to handle different lighting
                base_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                base_roi = cv2.equalizeHist(base_roi)
                
                # Multiple versions for robustness
                faces.append(base_roi)
                labels.append(label_map[name])
                faces.append(cv2.flip(base_roi, 1))
                labels.append(label_map[name])
                faces.append(cv2.convertScaleAbs(base_roi, alpha=0.8, beta=0)) # Darker
                labels.append(label_map[name])
                faces.append(cv2.convertScaleAbs(base_roi, alpha=1.2, beta=10)) # Brighter
                labels.append(label_map[name])

    # 2. Load from dataset/images_info.xlsx (New Dataset)
    xlsx_path = os.path.join(SCRIPT_DIR, 'dataset', 'images_info.xlsx')
    if os.path.exists(xlsx_path):
        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path)
            # Expecting 'image' and 'id' or 'caption' as name
            for _, row in df.iterrows():
                img_name = str(row.get('image', ''))
                if not img_name: continue
                # Search for image in 'dataset' folder
                found = False
                for ext in ['', '.png', '.jpg', '.jpeg']:
                    trial_path = os.path.join(SCRIPT_DIR, 'dataset', img_name + ext)
                    if os.path.exists(trial_path):
                        img = cv2.imread(trial_path)
                        if img is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            name = str(row.get('caption', 'Unknown')).split(' ')[0] # Use first word of caption as name
                            if name not in label_map: label_map[name] = len(label_map)
                            
                            faces.append(cv2.resize(gray, (100, 100)))
                            labels.append(label_map[name])
                            found = True
                            break
                if found: print(f'[SCAS] Loaded {img_name} from Excel dataset as {name}')
        except Exception as e:
            print(f'[SCAS] Excel Load Error: {e}')

    if faces:
        known_names = [''] * len(label_map)
        for name, lid in label_map.items(): known_names[lid] = name
        face_recognizer.train(faces, np.array(labels, dtype=np.int32))
        model_trained = True
        print(f'[SCAS] Trained on {len(label_map)} people matching: {list(label_map.keys())}')
    else:
        model_trained = False
load_known_faces()
try:
    pb = Pushbullet('o.r6ahTiaZNgGtTxjx6vFaUNgqYfKky1AU')
except Exception:
    pb = None
ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('blue')
scas_active = False
cap = None
out = None

class SCASApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('SCAS AI – Smart Camera Alert System')
        self.geometry('1100x650')
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.image_ref = None
        self.cleanup_complete = threading.Event()
        self.cleanup_complete.set()
        self._alert_cooldown = False
        self._face_history = []
        self.COINCIDENCE_THRESHOLD = 3
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=210, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky='nsew')
        self.sidebar.grid_rowconfigure(7, weight=1)
        ctk.CTkLabel(self.sidebar, text='SCAS AI', font=ctk.CTkFont(size=26, weight='bold')).grid(row=0, column=0, padx=20, pady=(20, 4))
        ctk.CTkLabel(self.sidebar, text='Smart Camera Alert System', font=ctk.CTkFont(size=10), text_color='gray').grid(row=1, column=0, padx=20, pady=(0, 10))
        self.status_label = ctk.CTkLabel(self.sidebar, text='● Inactive', text_color='#ff5555', font=ctk.CTkFont(size=13, weight='bold'))
        self.status_label.grid(row=2, column=0, padx=20, pady=6)
        self.activate_btn = ctk.CTkButton(self.sidebar, text='▶  Activate', fg_color='#1f8a1f', hover_color='#156315', command=self.start_activation_thread)
        self.activate_btn.grid(row=3, column=0, padx=20, pady=6)
        self.deactivate_btn = ctk.CTkButton(self.sidebar, text='■  Deactivate', fg_color='#8a1f1f', hover_color='#631515', command=self.deactivate_scas, state='disabled')
        self.deactivate_btn.grid(row=4, column=0, padx=20, pady=6)
        self.known_faces_btn = ctk.CTkButton(self.sidebar, text='👤  Manage Known Faces', command=self.manage_known_faces)
        self.known_faces_btn.grid(row=5, column=0, padx=20, pady=6)
        self.recordings_btn = ctk.CTkButton(self.sidebar, text='🎞  View Recordings', command=self.view_recordings)
        self.recordings_btn.grid(row=6, column=0, padx=20, pady=6)
        legend = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        legend.grid(row=8, column=0, padx=10, pady=(10, 20), sticky='s')
        ctk.CTkLabel(legend, text='Detection Legend', font=ctk.CTkFont(size=11, weight='bold'), text_color='gray').pack(anchor='w')
        ctk.CTkLabel(legend, text='🟩  Known Person – no alert', font=ctk.CTkFont(size=10), text_color='#55ff55').pack(anchor='w')
        ctk.CTkLabel(legend, text='🟥  Unknown Person – ALERT', font=ctk.CTkFont(size=10), text_color='#ff5555').pack(anchor='w')
        ctk.CTkLabel(legend, text='🟦  Object/Animal – no alert', font=ctk.CTkFont(size=10), text_color='#55aaff').pack(anchor='w')
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky='nsew')
        self.video_label = tk.Label(self.main_frame, text='Primary Feed (Inactive)', bg='#1a1a1a', fg='#888888', font=('Helvetica', 14))
        self.video_label.pack(expand=True, fill='both')
        self.alert_bar = ctk.CTkLabel(self, text='', fg_color='transparent', font=ctk.CTkFont(size=13, weight='bold'), text_color='#ff5555')
        self.alert_bar.grid(row=1, column=0, columnspan=2, sticky='ew', padx=20, pady=(0, 6))
        self.alarm_sound = None
        try:
            pygame.mixer.init()
            if os.path.exists(ALARM_PATH):
                self.alarm_sound = pygame.mixer.Sound(ALARM_PATH)
        except Exception:
            pass
        self.update_gui_loop()

    def update_gui_loop(self):
        try:
            if scas_active:
                with self.frame_lock:
                    frame_to_show = self.current_frame.copy() if self.current_frame is not None else None
                if frame_to_show is not None:
                    frame_to_show = cv2.resize(frame_to_show, (780, 520))
                    img_pil = Image.fromarray(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB))
                    self.image_ref = ImageTk.PhotoImage(image=img_pil)
                    self.video_label.configure(image=self.image_ref, text='')
        except Exception as e:
            print(f'[GUI] {e}')
        if self.winfo_exists():
            self.after(30, self.update_gui_loop)

    def start_activation_thread(self):
        if not self.cleanup_complete.is_set():
            messagebox.showinfo('Please Wait', 'System is still shutting down. Try again in a moment.')
            return
        self.activate_btn.configure(state='disabled')
        self.status_label.configure(text='● Initializing…', text_color='#ffaa00')
        self.cleanup_complete.clear()
        self.after(500, lambda: threading.Thread(target=self.activate_scas, daemon=True).start())

    def activate_scas(self):
        global scas_active, cap, out
        try:
            for idx in [0, 1]:
                cap = cv2.VideoCapture(idx)
                if cap and cap.isOpened():
                    break
            if not cap or not cap.isOpened():
                self.after(0, lambda: messagebox.showerror('Error', 'No camera found.'))
                self.after(0, self.reset_ui)
                self.cleanup_complete.set()
                return
            ts = int(pygame.time.get_ticks())
            record_file = os.path.join(RECORDINGS_DIR, f'scas_{ts}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(record_file, fourcc, 20.0, (640, 480))
            scas_active = True
            self.after(0, lambda: self.deactivate_btn.configure(state='normal'))
            self.after(0, lambda: self.status_label.configure(text='● Active', text_color='#55ff55'))
            self.detection_loop()
        except Exception as e:
            print(f'[SCAS] Activation error: {e}')
            self.after(0, self.reset_ui)
            self.cleanup_complete.set()

    def detection_loop(self):
        global scas_active, cap, out
        while scas_active:
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.run_detection(frame)
            if out:
                out.write(cv2.resize(annotated, (640, 480)))
            with self.frame_lock:
                self.current_frame = annotated.copy()
        self.cleanup()

    def run_detection(self, frame):
        if yolo_model is None:
            return frame
        
        # Run inference with YOLOv26
        results = yolo_model(frame, conf=0.4, verbose=False)
        
        unknown_detected = False
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                conf = float(box.conf[0])
                cid = int(box.cls[0])
                
                label_name = result.names[cid]
                
                if label_name == 'person':
                    recognized, person_name = self.identify_face(frame, x, y, w, h)
                    if recognized:
                        color = (50, 205, 50)
                        text = f'{person_name} ({conf:.0%})'
                    else:
                        color = (50, 50, 220)
                        text = f'Unknown  ALERT ({conf:.0%})'
                        unknown_detected = True
                else:
                    color = (210, 140, 30)
                    text = f'{label_name} ({conf:.0%})'
                
                # Draw box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 4, y), color, -1)
                cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
        if unknown_detected:
            self._trigger_alert()
        else:
            self._clear_alert()
        return frame

    def identify_face(self, frame, x, y, bw, bh):
        if not model_trained:
            return (False, '')
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + bw + pad)
        y2 = min(frame.shape[0], y + bh + pad)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return (False, '')
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_roi, 1.1, 5, minSize=(20, 20))
        
        best_match = None
        min_conf = float('inf')

        for fx, fy, fw, fh in faces:
            # Tighten bounding box exactly as in training
            ty = int(fy + 0.1 * fh)
            th = int(0.8 * fh)
            tx = int(fx + 0.1 * fw)
            tw = int(0.8 * fw)
            # Ensure boundaries are within roi
            if ty+th > gray_roi.shape[0] or tx+tw > gray_roi.shape[1]:
                continue
                
            face_img = cv2.resize(gray_roi[ty:ty + th, tx:tx + tw], (100, 100))
            face_img = cv2.equalizeHist(face_img)
            label, confidence = face_recognizer.predict(face_img)
            print(f'[SCAS] Face match debug: label={label}, confidence={confidence:.1f}')
            
            # Balanced threshold: 110 is a solid middle ground.
            # Below 100 was too strict (caused 'not detecting me' error).
            # Above 115 is too loose (caused 'everyone is me' error).
            if confidence < 110 and 0 <= label < len(known_names):
                if confidence < min_conf:
                    min_conf = confidence
                    best_match = known_names[label]

        if best_match:
            self._face_history.append(best_match)
        elif len(faces) > 0:
            self._face_history.append('Unknown')
        else:
            self._face_history.append('NoFace')
            
        # Keep only the last N frames of history (smooths out missed frames)
        if len(self._face_history) > 20: # store up to 20 frames
            self._face_history.pop(0)
            
        # Check if we recognized this known person reasonably often recently
        # E.g. 3 times in the last 15 frames
        recent_history = self._face_history[-15:]
        known_counts = {}
        for f in recent_history:
            if f not in ('Unknown', 'NoFace'):
                known_counts[f] = known_counts.get(f, 0) + 1
                
        best_known = None
        max_seen = 0
        for name, count in known_counts.items():
            if count >= self.COINCIDENCE_THRESHOLD and count > max_seen:
                best_known = name
                max_seen = count
                
        if best_known:
            return (True, best_known)
            
        return (False, '')

    def _trigger_alert(self):
        if not self._alert_cooldown:
            self._alert_cooldown = True
            self.after(0, lambda: self.alert_bar.configure(text='🚨  UNKNOWN PERSON DETECTED – INTRUDER ALERT!', fg_color='#6e1111'))
            if self.alarm_sound and (not pygame.mixer.get_busy()):
                self.alarm_sound.play(-1)
            threading.Thread(target=self._send_push, daemon=True).start()
            self.after(10000, self._reset_cooldown)

    def _reset_cooldown(self):
        self._alert_cooldown = False

    def _clear_alert(self):
        self._alert_cooldown = False
        if pygame.mixer.get_busy():
            pygame.mixer.stop()
        self.after(0, lambda: self.alert_bar.configure(text='', fg_color='transparent'))

    def _send_push(self):
        if pb:
            try:
                pb.push_note('🚨 SCAS ALERT', 'Unknown person detected by your camera!')
            except Exception:
                pass

    def deactivate_scas(self):
        global scas_active
        scas_active = False
        self.deactivate_btn.configure(state='disabled')
        self.status_label.configure(text='● Disconnecting…', text_color='#ffaa00')

    def cleanup(self):
        global cap, out
        try:
            if cap:
                cap.release()
                cap = None
            if out:
                out.release()
                out = None
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
        except Exception as e:
            print(f'[SCAS] Cleanup error: {e}')
        finally:
            self.after(500, self.finish_cleanup)

    def finish_cleanup(self):
        self.reset_ui()
        self.cleanup_complete.set()

    def reset_ui(self):
        self.activate_btn.configure(state='normal')
        self.deactivate_btn.configure(state='disabled')
        self.status_label.configure(text='● Inactive', text_color='#ff5555')
        self.alert_bar.configure(text='', fg_color='transparent')
        self.video_label.configure(image='', text='Primary Feed (Inactive)')
        with self.frame_lock:
            self.current_frame = None
            self.image_ref = None

    def manage_known_faces(self):
        os.startfile(KNOWN_FACES_DIR)
        messagebox.showinfo('Known Faces Folder', f'Add photos of trusted people here:\n{KNOWN_FACES_DIR}\n\n• Name each file after the person  (e.g. Keerthi.jpg)\n• Restart SCAS after adding new photos to retrain the model.\n\nSupported formats: .jpg  .jpeg  .png')

    def view_recordings(self):
        os.startfile(RECORDINGS_DIR)
if __name__ == '__main__':
    app = SCASApp()
    app.mainloop()