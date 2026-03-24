
import time
import numpy as np
from PIL import Image
import os
from pynput import mouse, keyboard # pip install pynput
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import threading
from queue import Queue
import mss
#ffmpeg
import av
import enum
import argparse


MAXSIZE_QUEUE = 300
DRAG_THRESHOLD = 15 #15 pixels
STREAM_LOOKAHEAD = 0.02 #20ms

class Action(enum.Enum):
    NONE = 0
    LEFT_CLICK = 1
    RIGHT_CLICK = 2
    DOUBLE_CLICK = 3
    DRAG_START = 4
    DRAG_END = 5
    SCROLL = 6
    KEY_PRESS = 7

orchestration_queue = Queue()
#this queue stores input capture requests.. as soon as frame taskes. screen shot

input_stream = Queue(maxsize=MAXSIZE_QUEUE) #our input stream queue (windowed there is max size)
#ill make sure that if we go beyond window we drop the end of queue abd add to front
#latest inputs are stored at end, oldest are at start.

class DataCollector:

    def __init__(self, session_name : str, fps : int, debug = False):
        #video state is file path of the 
        self.fps = fps #store fps controls how often we take screen shots etc
        self.session_path = Path("sessions") / session_name
        self.session_path.mkdir(parents=True, exist_ok=True)

        #have frame counter here
        self.frame_counter = 0 #static.
        self.running = False

        self.dragging = False
        self.mouse_down_position = None 
        self.mouse_down_ts = None
        self.is_mouse_down = False

        self.session_log = {}
        

        #debug flag
        self.debug = debug

    
    def _image_worker(self):
        #class emthod, that will be one of async wokrers.   
        with mss.mss() as sct:
            monitor = sct.monitors[0] #get the prmary montor
            while self.running:
                frame = sct.grab(monitor)
                #caputre frane right now.
                ts = time.perf_counter() #this is amonotinc counter, for performance sepcific tracking.

                #store file.
                img = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
                img.save(self.session_path / f"{self.session_path.name}{self.frame_counter:06d}.png")
                #put this in frame queue
                orchestration_queue.put({"ts" : ts, "frame_path": f"{self.session_path.name}{self.frame_counter:06d}.png" }) #might not need to send frame as well to queue?
                self.frame_counter += 1
                time.sleep(1 / self.fps)  # 10 fps for 10 frames a second.

    #i dont think this should be a sperate thread, this should just spawn the relevant pynput threads.
    #and we will be assigning no input based on if frame caputred, and we look in input stream and no input ast that timestamp
    #then assign that there was no input.

    def _input_worker(self):
        #set up the mouse and keyboard listeners
        def on_click(x, y, button, pressed):
            '''
            this listerner is porbably the most complex part
            if click then store into queue (with whatever button and xy)
            but if click and press down then store drag messages into queue with updated positions
            and on release then drag end no it should be if drag start (we measure drag by pixel difference between
            clikc initla xy and then pressed new postion) so one drag start (x,y) and when release we have a drag end
            no intermediate drag messages into queue.
            '''
            if pressed:
                #this is the first time on click will be fired. and then fired again on release.
                if button == mouse.Button.left:
                    self.is_mouse_down = True
                    ts = time.perf_counter()
                    #every time mouse is down, trac positon and ts
                    self.mouse_down_position = (x,y)
                    self.mouse_down_ts = ts
                    #do not log left clikc, since we dont know if it is a drag or not.

                elif button == mouse.Button.right:
                    ts = time.perf_counter()
                    input_stream.put({"ts" : ts, "action" : Action.RIGHT_CLICK, "x" : x, "y" : y})

            if not pressed:
                #here we check for drag stop, sicne on click is fired twice at each clikc (press and release)
                if button != mouse.Button.left:
                    #only left needs exra logic right is fine.
                    return

                if self.dragging:
                    #now drag stops
                    self.dragging = False
                    ts = time.perf_counter()
                    input_stream.put({"ts" : ts, "action" : Action.DRAG_END, "x" : x, "y" : y})
                else:
                    #it was a left clikc not a drag
                    input_stream.put({"ts" : self.mouse_down_ts, "action" : Action.LEFT_CLICK, "x" : x, "y" : y})

                self.is_mouse_down = False

        def on_move(x, y):
            if self.is_mouse_down:
                #calcuate last mouse down positon and current position if the delta is more than thresshold we are dragging
                #and then we can send in a drag action to input queue.
                dx = abs(x - self.mouse_down_position[0])
                dy = abs(y - self.mouse_down_position[1])

                if dx >= DRAG_THRESHOLD or dy >= DRAG_THRESHOLD:
                    #retroactively place this ts there.
                    #but queue isnt sorted? perhaps we must use a priority queue for this ?
                    #p - queue on ts field sort,(min heap) so to make sure we are always in order of ts.
                    #actually we dont need this, its a trade off instead of o logn heapifies every time anything is insereted
                    #this queue is high volume, will collapse. nobody cares for a 50 ms delay to be honest. of action and frame.

                    if not self.dragging: #only track drag once
                        ts = time.perf_counter()
                        input_stream.put({"ts" : ts, "action" : Action.DRAG_START, "x" : self.mouse_down_position[0], "y" : self.mouse_down_position[1]})
                        self.dragging = True

        def on_scroll(x, y, dx, dy):
            ts = time.perf_counter()
            input_stream.put({"ts": ts, "action": Action.SCROLL, "dx": dx, "dy": dy})

        def on_press(key):
            ts = time.perf_counter()
            try:
                char = key.char
            except AttributeError:
                char = str(key)  # special keys like Key.cmd, Key.delete
            input_stream.put({"ts": ts, "action": Action.KEY_PRESS, "key": char})


        #listeners, to spawn these threads.abs
        mouse_listener = mouse.Listener(on_click = on_click,
        on_move = on_move,
        on_scroll = on_scroll)

        keyboard_listener = keyboard.Listener(on_press=on_press)

        mouse_listener.start()
        keyboard_listener.start()

        while self.running:
            pass
        #only run this trhead till we "running" as soon as we stop, all thread must shut off.

        mouse_listener.stop()
        keyboard_listener.stop()


    def _orchestration_worker(self):
        '''
        this is our orchetartion thread, that should ingest from orchesatration queue
        like as soon a frame is released. and captured, then we ingest it, and find releavnt actions for this frame
        and generate a json action dict for that frame, and append to our file basically.
        and thats that..
        '''

        prev_frame = None
        prev_actions = None

        while self.running or not orchestration_queue.empty():
            try:
                latest_capture = orchestration_queue.get(timeout = 1)
                #get oldest, basically front of queue and wait till one second, if we dont wait we will quit loop immeditaly.
            except:
                continue
                
            #once we get the latest capture we get its timestmap to retireve the relavtn action 
            frame_ts = latest_capture["ts"]
            frame_image = latest_capture["frame_path"]

            actions = []

            while not input_stream.empty():
                event = input_stream.queue[0] #peek queue do not remove elemnt yet.

                if event["ts"] <= frame_ts + STREAM_LOOKAHEAD:
                    actions.append(input_stream.get())
                else:
                    break #break as soon as we go out of our input frame's time window

            if not actions:
                actions.append({"action" : Action.NONE})

            serialized_actions = []
            for a in actions:
                serialized_actions.append({
                    k: (v.name if isinstance(v, Action) else v)
                    for k, v in a.items()
                })


            #need to shift actions one frame behind to show causailty to our idm model.
            # assign current actions to PREVIOUS frame
            if prev_frame is not None:
                self.session_log[prev_frame] = serialized_actions

            # current frame gets previous actions (or NONE if first frame)
            if prev_frame is None:
                self.session_log[frame_image] = [{"action": "NONE"}]

            prev_frame = frame_image
            prev_actions = serialized_actions

            # flush last frame as NONE since no next frame to pull from
            if prev_frame is not None:
                self.session_log[prev_frame] = [{"action": "NONE"}]
        
    def _encode_in_mp4(self):
        frames = []
        #load all frames from path
        for f in os.listdir(self.session_path):
            if f.endswith(".png"):
                frames.append(f)
        
        if not frames:
            raise ValueError("No images found in the session path! You are cooked! rip")

        frames = sorted(frames) #sort by name (aka ts)
        width = 0
        height = 0
        first_frame = Image.open(self.session_path / frames[0]).convert('RGB')
        width, height = first_frame.size

        container = av.open(self.session_path / f"{self.session_path.name}.mp4", mode = "w")
        #create the container for this, now we need to attach the relevant video strema into this cpntainer.
        stream = container.add_stream(codec_name="h264", rate = self.fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'

        #now go through each frame send to ffmpeg and receive all packets and add in our video strema
        #in current container.abs
        for filename in frames:
            img = Image.open(self.session_path / filename).convert('RGB')
            #convert to a av Frame (video) 
            frame = av.VideoFrame.from_ndarray(np.array(img), format = "rgb24")

            for packet in stream.encode(frame):
                #receovbe untill stream stops producing packets (stream of this particular encodning)
                container.mux(packet) #done !

        #flush stream and end
        for packet in stream.encode():
            container.mux(packet)

        container.close()
        print(f"Training Video Stored at {self.session_path / f"{self.session_path.name}.mp4"}")
        return


    def _spawn_threads(self):
        self.running = True
        image_capture_thread = threading.Thread(target=self._image_worker, daemon=True)
        input_capture_thread = threading.Thread(target=self._input_worker, daemon=True)
        orchestration_thread = threading.Thread(target=self._orchestration_worker, daemon=True)

        image_capture_thread.start()
        input_capture_thread.start()
        orchestration_thread.start()

        return image_capture_thread, input_capture_thread, orchestration_queue

    def record_session(self):
        self._spawn_threads()
        print(f"Recording {self.session_path.name} — Ctrl+C to stop")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_session()
    
    def stop_session(self):
        print("Stopping...")
        self.running = False
        time.sleep(0.5)  # let threads flush remaining items

        self._build_session_json()
        self._encode_in_mp4()
        if not self.debug:
            self._cleanup_pngs()
        print(f"Done. Session at {self.session_path}")

    def _build_session_json(self):
        import json

        # sort by frame number to guarantee order
        ordered = dict(sorted(self.session_log.items()))

        out_path = self.session_path / f"{self.session_path.name}.json"
        with open(out_path, "w") as f:
            json.dump(ordered, f, indent=2)

        print(f"JSON saved: {out_path}")

    def _cleanup_pngs(self):
        for f in self.session_path.glob("*.png"):
            f.unlink()
        print("PNGs cleaned up")

    def pause_session(self):
        self.running = False

    def is_running(self):
        #return if running when we are closing it out need to be aware.
        return self.running


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI Workflow Data Collector")
    parser.add_argument("session_name", type=str, help="Name of the session e.g. session_001")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--debug", action="store_true", help="Keep PNGs after session for inspection")
    
    args = parser.parse_args()
    
    collector = DataCollector(session_name=args.session_name, fps=args.fps, debug=args.debug)
    collector.record_session()