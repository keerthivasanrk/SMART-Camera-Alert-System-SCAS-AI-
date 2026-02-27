import tkinter as tk
from tkinter import messagebox
import os
import cv2
import pygame
import numpy as np
from pushbullet import Pushbullet
import threading
import customtkinter as ctk
from PIL import Image, ImageTk

# Set path for the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

scas_active = False
cap = None
out = None

# Corrected paths
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "yolov3.weights")
CFG_PATH = os.path.join(SCRIPT_DIR, "yolov3.cfg")
ALARM_PATH = os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', '86502^alarm.wav') # Keep original but check existence
RECORDINGS_DIR = os.path.join(os.environ.get('USERPROFILE', ''), 'scas_recordings')

if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# Initialize YOLO
try:
    yolo_net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
except Exception as e:
    print(f"Error loading YOLO: {e}")

classes = ["person"]
pb = Pushbullet("o.r6ahTiaZNgGtTxjx6vFaUNgqYfKky1AU")

class SCASApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SCAS AI - Smart Camera Alert System")
        self.geometry("1000x600")

        # State management
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.image_ref = None # Persistent reference for Python's GC
        self.cleanup_complete = threading.Event()
        self.cleanup_complete.set()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SCAS AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Inactive", text_color="red")
        self.status_label.grid(row=1, column=0, padx=20, pady=10)

        self.activate_btn = ctk.CTkButton(self.sidebar_frame, text="Activate SCAS", command=self.start_activation_thread)
        self.activate_btn.grid(row=2, column=0, padx=20, pady=10)

        self.deactivate_btn = ctk.CTkButton(self.sidebar_frame, text="Deactivate SCAS", command=self.deactivate_scas, state="disabled")
        self.deactivate_btn.grid(row=3, column=0, padx=20, pady=10)

        self.recordings_btn = ctk.CTkButton(self.sidebar_frame, text="View Recordings", command=self.view_recordings)
        self.recordings_btn.grid(row=5, column=0, padx=20, pady=20)

        # Main View (Using standard tk.Label for high-perf video feed stability)
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.video_label = tk.Label(self.main_frame, text="Primary Feed (Inactive)", bg="#2b2b2b", fg="white")
        self.video_label.pack(expand=True, fill="both")

        self.alarm_sound = None
        try:
            pygame.mixer.init()
            if os.path.exists(ALARM_PATH):
                self.alarm_sound = pygame.mixer.Sound(ALARM_PATH)
        except: pass

        # Start GUI update loop
        self.update_gui_loop()

    def update_gui_loop(self):
        """Main thread loop for GUI updates"""
        try:
            if scas_active:
                frame_to_show = None
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_to_show = self.current_frame.copy()
                
                if frame_to_show is not None:
                    # Resize for consistency
                    frame_to_show = cv2.resize(frame_to_show, (640, 480))
                    img_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # Use ImageTk for standard tk.Label stability
                    self.image_ref = ImageTk.PhotoImage(image=img_pil)
                    self.video_label.configure(image=self.image_ref, text="")
        except Exception as e:
            print(f"GUI Sync Error: {e}")
        
        # Schedule next update
        if self.winfo_exists():
            self.after(33, self.update_gui_loop)

    def start_activation_thread(self):
        if not self.cleanup_complete.is_set():
            messagebox.showinfo("Wait", "System is still deactivating. Please wait a moment.")
            return
        
        self.activate_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Initializing...", text_color="yellow")
        self.cleanup_complete.clear()
        
        # Small delay to let OS release camera handle
        self.after(500, lambda: threading.Thread(target=self.activate_scas, daemon=True).start())

    def activate_scas(self):
        global scas_active, cap, out
        try:
            # Try camera 0 then 1
            for i in [0, 1]:
                cap = cv2.VideoCapture(i)
                if cap and cap.isOpened(): break
            
            if not cap or not cap.isOpened():
                self.after(0, lambda: messagebox.showerror("Error", "No camera detected. Check connection."))
                self.after(0, self.reset_ui)
                self.cleanup_complete.set()
                return

            # Dynamic filename for recording
            timestamp = int(pygame.time.get_ticks())
            record_file = os.path.join(RECORDINGS_DIR, f'scas_event_{timestamp}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(record_file, fourcc, 20.0, (640, 480))

            scas_active = True
            self.after(0, lambda: self.deactivate_btn.configure(state="normal"))
            self.after(0, lambda: self.status_label.configure(text="Status: Active", text_color="green"))
            
            self.detection_loop()
        except Exception as e:
            print(f"Activation Failed: {e}")
            self.after(0, self.reset_ui)
            self.cleanup_complete.set()

    def detection_loop(self):
        global scas_active, cap, out
        person_detected_flag = False
        
        while scas_active:
            if cap is None: break
            ret, frame = cap.read()
            if not ret: break

            # Detection Logic
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            outputs = yolo_net.forward(output_layers)

            person_detected = False
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.4 and class_id == 0:
                        person_detected = True
                        h, w = frame.shape[:2]
                        cx, cy = int(detection[0]*w), int(detection[1]*h)
                        dw, dh = int(detection[2]*w), int(detection[3]*h)
                        x, y = int(cx - dw/2), int(cy - dh/2)
                        cv2.rectangle(frame, (x, y), (x+dw, y+dh), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Alerts
            if person_detected:
                if not person_detected_flag:
                    try: pb.push_note("Alert", "Activity detected!")
                    except: pass
                    person_detected_flag = True
                if self.alarm_sound and not pygame.mixer.get_busy():
                    self.alarm_sound.play(-1)
            else:
                if pygame.mixer.get_busy(): pygame.mixer.stop()
                person_detected_flag = False

            if out: out.write(frame)
            
            with self.frame_lock:
                self.current_frame = frame.copy()

        self.cleanup()

    def deactivate_scas(self):
        global scas_active
        scas_active = False
        self.deactivate_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Disconnecting...", text_color="orange")

    def cleanup(self):
        global cap, out
        try:
            if cap: cap.release(); cap = None
            if out: out.release(); out = None
            if pygame.mixer.get_busy(): pygame.mixer.stop()
        except Exception as e:
            print(f"Hardware Release Error: {e}")
        finally:
            self.after(500, self.finish_cleanup)

    def finish_cleanup(self):
        self.reset_ui()
        self.cleanup_complete.set()

    def reset_ui(self):
        self.activate_btn.configure(state="normal")
        self.deactivate_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Inactive", text_color="red")
        self.video_label.configure(image="", text="Primary Feed (Inactive)")
        with self.frame_lock:
            self.current_frame = None
            self.image_ref = None

    def view_recordings(self):
        os.startfile(RECORDINGS_DIR)

if __name__ == "__main__":
    app = SCASApp()
    app.mainloop()
