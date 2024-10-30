import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

#ทำหน้าที่แปลงภาพสีให้เป็นภาพไบนารี โดยผ่านการแปลงเป็นสีเทา
def binarize_image(image, threshold=150):
    grayscale_image = image.convert('L')   # แปลงภาพเป็นสีเทา (Grayscale)
    gray_array = np.array(grayscale_image)  # แปลงภาพสีเทาเป็น Array ของ numpy

    # ใช้ Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_array, (5, 5), 0)
    
    # เพิ่มความเข้มของภาพเพื่อเน้นพิกเซลสีดำและขาว
    enhanced_array = np.clip(blurred_image * 1.5, 0, 255).astype(np.uint8)
    binary_array = np.array(enhanced_array) > threshold
    binary_image = Image.fromarray(binary_array.astype(np.uint8) * 255)# คูณ 255 เพื่อให้ภาพเป็นขาวดำ
    return binary_image # ส่งกลับภาพไบนารี

#คำนวณ Vertical Projection ให้ข้อมูลว่ามีพิกเซลสีดำอยู่ในแต่ละคอลัมน์เท่าไหร่
# 
def vertical_projection(binary_image):
    # แปลงภาพไบนารีเป็น Array ของ numpy
    img_array = np.array(binary_image) 
    # นับจำนวนพิกเซลสีดำในแต่ละคอลัมน์
    projection = np.sum(img_array == 0, axis=0)
    return projection # ส่งกลับ array ของ Vertical Projection

# ฟังก์ชันแยกขอบเขตตัวอักษรจาก Vertical Projection
def segment_characters_modified(projection, min_width=5, threshold=5):
    segments = [] # สร้าง list
    in_character = False # ตัวแปรบอกสถานะว่ากำลังอยู่ในตัวอักษรหรือไม่
    start_index = 0 # ตำแหน่งเริ่มต้นของตัวอักษร
    
    for i, value in enumerate(projection): # วนลูปผ่านค่าใน Projection
        if value > threshold and not in_character:# ถ้าเจอพิกเซลสีดำและยังไม่อยู่ในตัวอักษร
            in_character = True # ตั้งค่าให้อยู่ในตัวอักษร
            start_index = i# บันทึกตำแหน่งเริ่มต้น
            
            # ถ้าหลุดจากตัวอักษรหรือถึงคอลัมน์สุดท้าย
        elif (value <= threshold and in_character) or (i == len(projection)-1 and in_character):
            in_character = False
            end_index = i if value <= threshold else i+1# บันทึกตำแหน่งสิ้นสุด
            if end_index - start_index >= min_width:
                segments.append((start_index, end_index)) # เก็บช่วงใน list segments
    return segments # ส่งกลับ list ของขอบเขตตัวอักษร

#ค่า min_width ใช้กรองช่วงตัวอักษรที่เล็กเกินไปหรือไม่สมบูรณ์ 
# ซึ่งอาจเป็น noise หรือรอยเล็กๆ ที่ไม่ใช่ตัวอักษรจริงๆ 
# หากขนาดของช่วงตัวอักษรน้อยกว่า min_width จะไม่ถูกบันทึกไว้ใน segments ครับ

# ฟังก์ชันดึงตัวอักษรจากภาพตามขอบเขตที่ได้
def extract_characters(binary_image, segments):
    # แปลงภาพไบนารีเป็น Array ของ numpy
    img_array = np.array(binary_image) 
    characters = []  # สร้าง list เพื่อเก็บตัวอักษรแต่ละตัว
    # วนลูปผ่านแต่ละขอบเขต
    for start, end in segments:
        if start >= 0 and end <= img_array.shape[1]:
            char_img = img_array[:, start:end] # ตัดภาพตัวอักษรตามขอบเขต
            rows = np.any(char_img == 0, axis=1) # เลือกแถวที่มีพิกเซลสีดำ
            char_img = char_img[rows]  # ตัดแถวที่ไม่มีพิกเซลสีดำออก
            characters.append(char_img) # เก็บภาพตัวอักษรใน list characters
    return characters
# np.any(char_img == 0, axis=1) ใช้เพื่อตรวจสอบว่ามีพิกเซลสีดำในแถวนั้นหรือไม่
#ถ้าไม่มีพิกเซลสีดำในแถวนั้นจะถือว่าเป็นช่องว่างและลบแถวนั้นออก


# ฟังก์ชันเตรียม pattern ของตัวอักษรเป้าหมายจากไฟล์ภาพ
def get_character_pattern(image_path, target_size=(40, 40)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # อ่านภาพและแปลงเป็น Grayscale
    resized_image = cv2.resize(  # resize ภาพให้มีขนาด 40x40
        image, target_size, interpolation=cv2.INTER_AREA) # แปลงเป็นภาพไบนารีแบบ Inverse
    _, binary_image = cv2.threshold( 
        resized_image, 127, 255, cv2.THRESH_BINARY_INV) 
    vertical_projection = np.sum(binary_image, axis=0) # คำนวณ Vertical Projection
    pattern = vertical_projection / max(vertical_projection)# normalize ค่าใน pattern
    return pattern

# ฟังก์ชันโหลด pattern ของตัวอักษรต่างๆ จากไฟล์
def load_character_patterns():
    char_images = {
        '0': 'D:\\KUKPS\\ComVision\\Project\\Number\\0.jpg',
        '1': 'D:\\KUKPS\\ComVision\\Project\\Number\\1.jpg',
        '2': 'D:\\KUKPS\\ComVision\\Project\\Number\\2.jpg',
        '3': 'D:\\KUKPS\\ComVision\\Project\\Number\\3.jpg',
        '4': 'D:\\KUKPS\\ComVision\\Project\\Number\\4.jpg',
        '5': 'D:\\KUKPS\\ComVision\\Project\\Number\\5.jpg',
        '6': 'D:\\KUKPS\\ComVision\\Project\\Number\\6.jpg',
        '7': 'D:\\KUKPS\\ComVision\\Project\\Number\\7.jpg',
        '8': 'D:\\KUKPS\\ComVision\\Project\\Number\\8.jpg',
        '9': 'D:\\KUKPS\\ComVision\\Project\\Number\\9.jpg',

        'ก': 'D:\\KUKPS\\ComVision\\Project\\Letter\\01.jpg',
        'ข': 'D:\\KUKPS\\ComVision\\Project\\Letter\\02.jpg',
        'ฃ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\03.jpg',
        'ค': 'D:\\KUKPS\\ComVision\\Project\\Letter\\04.jpg',
        'ฅ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\05.jpg',
        'ฆ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\06.jpg',
        'ง': 'D:\\KUKPS\\ComVision\\Project\\Letter\\07.jpg',
        'จ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\08.jpg',
        'ฉ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\09.jpg',
        'ช': 'D:\\KUKPS\\ComVision\\Project\\Letter\\10.jpg',
        'ซ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\11.jpg',
        'ฌ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\12.jpg',
        'ญ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\13.jpg',
        'ฎ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\14.jpg',
        'ฏ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\15.jpg',
        'ฐ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\16.jpg',
        'ฑ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\17.jpg',
        'ฒ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\18.jpg',
        'ณ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\19.jpg',
        'ด': 'D:\\KUKPS\\ComVision\\Project\\Letter\\20.jpg',
        'ต': 'D:\\KUKPS\\ComVision\\Project\\Letter\\21.jpg',
        'ถ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\22.jpg',
        'ท': 'D:\\KUKPS\\ComVision\\Project\\Letter\\23.jpg',
        'ธ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\24.jpg',
        'น': 'D:\\KUKPS\\ComVision\\Project\\Letter\\25.jpg',
        'บ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\26.jpg',
        'ป': 'D:\\KUKPS\\ComVision\\Project\\Letter\\27.jpg',
        'ผ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\28.jpg',
        'ฝ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\29.jpg',
        'พ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\30.jpg',
        'ฟ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\31.jpg',
        'ภ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\32.jpg',
        'ม': 'D:\\KUKPS\\ComVision\\Project\\Letter\\33.jpg',
        'ย': 'D:\\KUKPS\\ComVision\\Project\\Letter\\34.jpg',
        'ร': 'D:\\KUKPS\\ComVision\\Project\\Letter\\35.jpg',
        'ล': 'D:\\KUKPS\\ComVision\\Project\\Letter\\36.jpg',
        'ว': 'D:\\KUKPS\\ComVision\\Project\\Letter\\37.jpg',
        'ศ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\38.jpg',
        'ษ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\39.jpg',
        'ส': 'D:\\KUKPS\\ComVision\\Project\\Letter\\40.jpg',
        'ห': 'D:\\KUKPS\\ComVision\\Project\\Letter\\41.jpg',
        'ฬ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\42.jpg',
        'อ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\43.jpg',
        'ฮ': 'D:\\KUKPS\\ComVision\\Project\\Letter\\44.jpg'

    }
    patterns = {}  # เก็บ pattern
    for char, path in char_images.items(): # วนลูปผ่านแต่ละตัวอักษร
        patterns[char] = get_character_pattern(path)  # สร้าง pattern และเก็บใน dictionary
    return patterns

# ฟังก์ชันรู้จำตัวอักษรในภาพโดยเทียบกับ pattern

def recognize_characters(projection, segments, char_patterns):
    results = [] # list เพื่อเก็บผลลัพธ์การรู้จำ
    threshold = 0.5  # Threshold for matching
    
    # วนลูปผ่านขอบเขตตัวอักษรแต่ละตัว
    for start, end in segments:
        char_projection = projection[start:end]  # คำนวณ projection
        # ถ้าตัวอักษรสั้นเกินไป ให้ข้ามไป
        if len(char_projection) < 2:
            continue

        # normalize และ smooth ค่า projection ของตัวอักษร
        normalized_projection = np.interp(
            np.linspace(0, 1, 40),
            np.linspace(0, 1, len(char_projection)),
            char_projection
        )
        
        normalized_projection = normalized_projection / \
            max(normalized_projection)

        # คำนวณความคล้ายกับ pattern ของตัวอักษรแต่ละตัว
        similarities = {}
        for char, pattern in char_patterns.items():
            similarity = np.sum(np.abs(normalized_projection - pattern))
            similarities[char] = similarity

        # เลือกตัวอักษรที่มีความคล้ายมากที่สุด
        best_match = min(similarities, key=similarities.get)

        # Check if best match is within the threshold ตรวจสอบว่ารับได้ไหมกับค่า T
        if similarities[best_match] < threshold:
            results.append(best_match)
        else:
            # Estimate close matches เลือกใกล้เคียงสุด
            close_matches = sorted(similarities.items(), key=lambda x: x[1])
            estimated_match = close_matches[0][0]  # Best guess เลือกใกล้เคียงสุด
            results.append(estimated_match)  # เพิ่มผลลัพธ์ที่เดาไว้

    return results

# ฟังก์ชันหลักในการประมวลผลภาพ
def process_image(image_path):
    # font
    plt.rcParams['font.family'] = 'Tahoma'

    char_patterns = load_character_patterns()

    normalized_path = os.path.normpath(image_path)# จัดการเส้นทางไฟล์ภาพให้เป็นปกติ
    #เช็คว่ามีไฟล์ไหม
    if not os.path.exists(normalized_path):
        print(f"Image file not found: {normalized_path}")
        return

    # เปิดภาพจากเส้นทางที่ระบุ
    image = Image.open(normalized_path)
    new_size = (633, 131) # กำหนดขนาดใหม่สำหรับภาพ
    image = image.resize(new_size)  # ปรับขนาดภาพ

    binary_image = binarize_image(image, threshold=100)# แปลงภาพเป็นไบนารี
    projection = vertical_projection(binary_image)  # คำนวณ Vertical Projection ของภาพ
    # แสดง Vertical Projection
    plt.figure(figsize=(12, 4))
    plt.plot(projection)
    plt.title("Vertical Projection")
    plt.xlabel('Column Index')
    plt.ylabel('Black Pixel Count')
    plt.grid(True)
    plt.show()

    segments = segment_characters_modified(projection)  # แยกขอบเขตของตัวอักษร
    characters = extract_characters(binary_image, segments) # ดึงตัวอักษรแต่ละตัวจากภาพ
    recognized = recognize_characters(projection, segments, char_patterns)#จำตัวอักษร
    result_text = ''.join(recognized).strip()# รวมผลลัพธ์ตัวอักษรทั้งหมดเป็นข้อความ
    print("Recognized text:", result_text)# แสดงข้อความ

    # แสดงตัวอักษรแต่ละตัว
    if len(characters) > 0: # กำหนดจำนวนคอลัมน์ในการแสดงตัวอักษร
        cols = min(5, len(characters)) # คำนวณจำนวนแถว
        rows = (len(characters) - 1) // cols + 1
        plt.figure(figsize=(cols * 2, rows * 2))
        for i, char_img in enumerate(characters):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(char_img, cmap='gray')
            plt.title(f'Char: {recognized[i] if i < len(recognized) else ""}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        # ถ้าไม่พบตัวอักษรจะแสดงข้อความนี้
        print("No characters were segmented.")


if __name__ == "__main__":
    test01_path = "D:\\KUKPS\\ComVision\\Project\\test01.jpg"
    process_image(test01_path)
    test02_path = "D:\\KUKPS\\ComVision\\Project\\test02.jpg"
    process_image(test02_path)
    test03_path = "D:\\KUKPS\\ComVision\\Project\\test03.jpg"
    process_image(test03_path)
    test04_path = "D:\\KUKPS\\ComVision\\Project\\test04.jpg"
    process_image(test04_path)

#โปรแกรมเริ่มต้นในส่วน main ซึ่งจะกำหนดเส้นทางไปยังภาพทดสอบทั้ง 4 ภาพ 
# (ในโฟลเดอร์ที่กำหนด) จากนั้นเรียกใช้ฟังก์ชัน process_image เพื่อประมวลผลแต่ละภาพ

# เรียกใช้ฟังก์ชัน process_image

#เรียกใช้ load_character_patterns() เพื่อโหลด pattern ของตัวอักษรที่เตรียมไว้ 
# (ใช้สำหรับการรู้จำตัวอักษร)
# แปลงภาพและทำการประมวลผล แปลงภาพเป็นไบนารีโดยใช้ binarize_image() โดยให้ threshold ที่ 100 เพื่อทำให้ภาพเป็นขาว-ดำ
#สร้าง Vertical Projection โดยใช้ vertical_projection() ซึ่งจะให้ array 
# ที่บอกจำนวนพิกเซลสีดำในแต่ละคอลัมน์
#แสดง Vertical Projectionเพื่อให้เห็นว่าแต่ละคอลัมน์มีพิกเซลสีดำเท่าไหร่
#แยกขอบเขตของตัวอักษร ใช้ segment_characters_modified() 
# เพื่อหาขอบเขตของตัวอักษรจาก Vertical Projection
#เก็บตัวอักษรใช้ recognize_characters() เพื่อรู้จำตัวอักษรโดยเทียบ projection 
# ของตัวอักษรแต่ละตัวกับ pattern ที่โหลดมา


#binarize_image(): แปลงภาพเป็นภาพไบนารี 
# โดยแปลงเป็นสีเทาก่อน จากนั้นทำการเบลอและปรับความเข้ม 
# แล้วแปลงเป็นไบนารีโดยใช้ threshold

#vertical_projection(): คำนวณจำนวนพิกเซลสีดำในแต่ละคอลัมน์ของภาพไบนารี

#segment_characters_modified(): 
# หาขอบเขตของตัวอักษรจาก Vertical Projection โดยใช้ threshold 
# เพื่อตรวจจับช่วงที่เป็นตัวอักษร

#extract_characters(): ดึงตัวอักษรแต่ละตัวออกมาตามขอบเขตที่พบ

#get_character_pattern(): เตรียม pattern ของตัวอักษรจากไฟล์ภาพ 
# โดยคำนวณ Vertical Projection ของภาพแต่ละตัวอักษร

#load_character_patterns(): โหลด pattern ของตัวอักษรจากไฟล์ภาพที่เตรียมไว้

#recognize_characters(): เปรียบเทียบตัวอักษรที่แยกได้กับ pattern ที่เตรียมไว้ 
# เพื่อตรวจจับตัวอักษรที่มีความคล้ายมากที่สุด

#โดยรวม โค้ดนี้อ่านภาพ แปลงเป็นไบนารี แยกตัวอักษรในภาพ จากนั้นรู้จำตัวอักษรแต่ละตัวด้วยการเทียบ pattern ที่เตรียมไว้ สุดท้ายแสดงผลตัวอักษรที่ตรวจพบ
