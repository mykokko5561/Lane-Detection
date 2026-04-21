<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/e37fff78-8cd9-4237-9a89-a0c0269494d5" /># Lane-Detection  (Eng.)

This project implements a foundational computer vision pipeline for lane detection, a core component of autonomous vehicle navigation. Using OpenCV, the system applies a perspective transformation to generate a bird's-eye view of the road, isolates lane markings through dynamic HSV color thresholding, and identifies lane bases using histogram analysis. Finally, it utilizes the classic Sliding Window algorithm to track the lane lines frame-by-frame, projecting the detected safe driving path back onto the original video stream. It serves as a practical demonstration of image processing techniques, matrix transformations, and algorithmic thinking in robotics.
 
# Şerit-Takibi (Türkçe)

Bu proje, otonom araç navigasyonunun temel bileşenlerinden biri olan şerit tespiti için OpenCV tabanlı bir bilgisayarlı görü (computer vision) boru hattı (pipeline) sunmaktadır. Sistem, yola kuşbakışı bir açıdan bakmak için perspektif dönüşümü uygular, HSV renk eşikleme ile şerit çizgilerini arka plandan ayırır ve histogram analizi ile şeritlerin başlangıç noktalarını belirler. Ardından, klasik Kayan Pencere (Sliding Window) algoritmasını kullanarak şeritleri kare kare takip eder ve hesaplanan güvenli sürüş rotasını orijinal video akışının üzerine yansıtır. Görüntü işleme tekniklerinin, matris dönüşümlerinin ve robotik alanındaki algoritmik düşünce yapısının pratik bir gösterimi olarak tasarlanmıştır.

# The Sliding Window Algorithm (Eng.)

The Sliding Window algorithm serves as the core tracking mechanism of this project. After applying the perspective transformation and color masking, the system computes a pixel histogram of the lower half of the image to identify the starting x-coordinates of the left and right lanes. From these base points, a series of rectangular "windows" sequentially slide upward across the image, isolating the non-zero pixels that represent the lane markings. By calculating the center of mass (mean position) for the pixels within each window, the algorithm dynamically adjusts the horizontal position of the subsequent window. This continuous recentering allows the system to accurately track and trace the lane lines, adapting to curves and changes in the road's trajectory.

# Kayan Pencere Algoritması (Türkçe)

Kayan Pencere algoritması, bu projenin temel şerit takip mekanizması olarak görev yapmaktadır. Perspektif dönüşümü ve renk maskeleme işlemlerinin ardından sistem, sol ve sağ şeritlerin başlangıç x-koordinatlarını tespit etmek için görüntünün alt yarısının piksel histogramını hesaplar. Bu temel noktalardan başlayarak, bir dizi dikdörtgen "pencere" görüntü boyunca sıralı olarak yukarı doğru kayar ve şerit çizgilerini temsil eden pikselleri izole eder. Her pencere içindeki piksellerin kütle merkezini (ortalama konumunu) hesaplayan algoritma, bir sonraki pencerenin yatay konumunu dinamik olarak günceller. Bu sürekli merkezleme işlemi, sistemin yolun kıvrımlarına ve yörünge değişikliklerine adapte olarak şerit çizgilerini yüksek hassasiyetle takip etmesini sağlar.



<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/bd844b19-2dfb-4605-93a2-f3a8fa63ee9a" />








