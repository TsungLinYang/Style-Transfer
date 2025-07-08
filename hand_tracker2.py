import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, max_num_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def is_open_hand(self, frame, y_threshold=0.02):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
    
        if not results.multi_hand_landmarks:
            return False
    
        # 只檢查第一隻手（可擴充支援雙手）
        hand_landmarks = results.multi_hand_landmarks[0]

        finger_tip_ids = [8, 12, 16, 20]  # 食指、中指、無名指、小指
        finger_base_ids = [6, 10, 14, 18]
    
        open_fingers = 0
    
        for tip_id, base_id in zip(finger_tip_ids, finger_base_ids):
            tip_y = hand_landmarks.landmark[tip_id].y
            base_y = hand_landmarks.landmark[base_id].y
    
            # 加入 y 軸閾值，避免手微微彎曲時仍判定為張開
            if base_y - tip_y > y_threshold:
                open_fingers += 1
    
        # 判斷是否至少有 4 根手指張開
        return open_fingers >= 4


    def is_scissor_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # 食指(8)、中指(12) 伸直，其餘（無名指16、小指20）彎曲
            extended = []
            folded = []

            for tip, base in zip([8, 12], [6, 10]):
                extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y)

            for tip, base in zip([16, 20], [14, 18]):
                folded.append(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y)

            if all(extended) and all(folded):
                return True
        return False


    def close(self):
        self.hands.close()
