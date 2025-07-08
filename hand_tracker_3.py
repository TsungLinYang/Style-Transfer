import mediapipe as mp
import cv2
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
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

    # 真剪刀
    # def is_scissor_hand(self, frame):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(frame_rgb)

    #     if results.multi_hand_landmarks:
    #         hand_landmarks = results.multi_hand_landmarks[0]

    #         # 食指(8)、中指(12) 伸直，其餘（無名指16、小指20）彎曲
    #         extended = []
    #         folded = []

    #         for tip, base in zip([8, 12], [6, 10]):
    #             extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y)

    #         for tip, base in zip([16, 20], [14, 18]):
    #             folded.append(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y)

    #         if all(extended) and all(folded):
    #             return True
    #     return False
    
    # 真剪刀2
    def is_scissor_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # 限制必須只有一隻手才可以觸發剪刀手勢
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 1:
            return False

        hand_landmarks = results.multi_hand_landmarks[0]

        # 食指與中指伸直，其餘手指彎曲
        extended = []
        folded = []

        for tip, base in zip([8, 12], [6, 10]):
            extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y)

        for tip, base in zip([16, 20], [14, 18]):
            folded.append(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y)

        if all(extended) and all(folded):
            return True
        return False
    
    # 石頭
    # def is_scissor_hand(self, frame):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(frame_rgb)

    #     if results.multi_hand_landmarks:
    #         hand_landmarks = results.multi_hand_landmarks[0]

    #         folded = []

    #         # 除了拇指的四指 tip 與 base 對照
    #         for tip, base in zip([8, 12, 16, 20], [6, 10, 14, 18]):
    #             # tip 在 base 的下方（y值大） → 彎曲
    #             folded.append(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y)

    #         # 可選加上拇指判斷（不是必要）
    #         thumb_folded = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x

    #         if all(folded):  # 你也可以加上 thumb_folded 條件
    #             return True
    #     return False
    
    # def is_cross_finger_gesture(self, frame):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(frame_rgb)
    
    #     if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
    #         return False  # 至少需要兩隻手
    
    #     # 取出兩隻手
    #     hand1 = results.multi_hand_landmarks[0]
    #     hand2 = results.multi_hand_landmarks[1]
    
    #     def is_scissor_like(hand):
    #         # 只伸出 index (8) & middle (12)，其他 fingers 皆彎曲
    #         extended = []
    #         for tip, base in zip([8, 12, 16, 20], [6, 10, 14, 18]):
    #             if hand.landmark[tip].y < hand.landmark[base].y:
    #                 extended.append(tip)
    #         return set(extended) == {8, 12}  # 僅食指中指
    
    #     # 檢查兩隻手是否都呈剪刀狀
    #     if not (is_scissor_like(hand1) and is_scissor_like(hand2)):
    #         return False
    
    #     # 檢查食指與中指之間是否靠近（例如 index(8) 的距離 < 某個 threshold）
    #     idx1 = np.array([hand1.landmark[8].x, hand1.landmark[8].y])
    #     idx2 = np.array([hand2.landmark[8].x, hand2.landmark[8].y])
    #     mid1 = np.array([hand1.landmark[12].x, hand1.landmark[12].y])
    #     mid2 = np.array([hand2.landmark[12].x, hand2.landmark[12].y])
    
    #     dist_index = np.linalg.norm(idx1 - idx2)
    #     dist_middle = np.linalg.norm(mid1 - mid2)
    
    #     # threshold 可調整（0.1 在 normalized space 約等於手指寬度）
    #     return dist_index < 0.1 and dist_middle < 0.1

    # def is_cross_finger_gesture(self, frame):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(frame_rgb)

    #     if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
    #         return False

    #     hand1 = results.multi_hand_landmarks[0]
    #     hand2 = results.multi_hand_landmarks[1]

    #     def is_scissor_like(hand):
    #         tips = [8, 12]
    #         bases = [6, 10]
    #         extended = []
    #         for tip, base in zip(tips, bases):
    #             if hand.landmark[tip].y < hand.landmark[base].y:
    #                 extended.append(tip)
    #         return set(extended) == {8, 12}

    #     if not (is_scissor_like(hand1) and is_scissor_like(hand2)):
    #         return False

    #     # 提取兩手指尖位置
    #     points = [
    #         np.array([hand1.landmark[8].x, hand1.landmark[8].y]),   # hand1 index
    #         np.array([hand1.landmark[12].x, hand1.landmark[12].y]),  # hand1 middle
    #         np.array([hand2.landmark[8].x, hand2.landmark[8].y]),   # hand2 index
    #         np.array([hand2.landmark[12].x, hand2.landmark[12].y])  # hand2 middle
    #     ]

    #     # 檢查任意兩點之間是否都非常接近（表示幾乎重疊）
    #     threshold = 0.04  # 可以調整靈敏度（4% 螢幕寬度）
    #     close_count = 0

    #     for i in range(len(points)):
    #         for j in range(i + 1, len(points)):
    #             dist = np.linalg.norm(points[i] - points[j])
    #             if dist < threshold:
    #                 close_count += 1

    #     # 如果四指中有 3 對以上都非常靠近，代表手指重疊交叉
    #     return close_count >= 3
    
    

    def is_cross_finger_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
            return False

        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]

        # 確認每隻手是否有伸出食指與中指
        def has_index_middle_extended(hand):
            is_index_extended = hand.landmark[8].y < hand.landmark[6].y
            is_middle_extended = hand.landmark[12].y < hand.landmark[10].y
            return is_index_extended and is_middle_extended

        if not (has_index_middle_extended(hand1) and has_index_middle_extended(hand2)):
            return False

        # 取手腕與中指向量
        def get_direction_vector(hand):
            wrist = np.array([hand.landmark[0].x, hand.landmark[0].y])
            middle_tip = np.array([hand.landmark[12].x, hand.landmark[12].y])
            return middle_tip - wrist

        v1 = get_direction_vector(hand1)
        v2 = get_direction_vector(hand2)

        # 判斷是否近似垂直（十字）
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        is_cross_like = abs(cos_angle) < 0.3

        return is_cross_like
    
    # def is_cross_finger_gesture(self, frame):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.hands.process(frame_rgb)

    #     if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
    #         return False  # 至少需要兩隻手

    #     hand1 = results.multi_hand_landmarks[0]
    #     hand2 = results.multi_hand_landmarks[1]

    #     def is_scissor_like(hand):
    #         tips = [8, 12]
    #         bases = [6, 10]
    #         extended = []
    #         for tip, base in zip(tips, bases):
    #             if hand.landmark[tip].y < hand.landmark[base].y:
    #                 extended.append(tip)
    #         return set(extended) == {8, 12}  # 僅食指與中指伸直

    #     if not (is_scissor_like(hand1) and is_scissor_like(hand2)):
    #         return False

    #     # 取兩隻手的指尖位置（normalized coordinates）
    #     idx1 = np.array([hand1.landmark[8].x, hand1.landmark[8].y])
    #     mid1 = np.array([hand1.landmark[12].x, hand1.landmark[12].y])
    #     idx2 = np.array([hand2.landmark[8].x, hand2.landmark[8].y])
    #     mid2 = np.array([hand2.landmark[12].x, hand2.landmark[12].y])

    #     # 計算每對指尖之間的距離
    #     dist_ii = np.linalg.norm(idx1 - idx2)
    #     dist_mm = np.linalg.norm(mid1 - mid2)
    #     dist_cross1 = np.linalg.norm(idx1 - mid2)
    #     dist_cross2 = np.linalg.norm(mid1 - idx2)

    #     # 判定是否有交叉併攏（你可微調 threshold）
    #     close_threshold = 0.06  # 6% 畫面寬度以內算併攏

    #     is_crossed = (
    #         (dist_ii < close_threshold and dist_mm < close_threshold) or
    #         (dist_cross1 < close_threshold and dist_cross2 < close_threshold)
    #     )

    #     return is_crossed

    def close(self):
        self.hands.close()
