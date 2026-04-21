# Lane-Detection (Eng.)

This project implements a foundational computer vision pipeline for lane detection, a core component of autonomous vehicle navigation. Using OpenCV, the system applies a perspective transformation to generate a bird's-eye view of the road, isolates lane markings through dynamic HSV color thresholding, and identifies lane bases using histogram analysis. Finally, it utilizes the classic Sliding Window algorithm to track the lane lines frame-by-frame, projecting the detected safe driving path back onto the original video stream. It serves as a practical demonstration of image processing techniques, matrix transformations, and algorithmic thinking in robotics.
 
# Şerit-Takibi (Türkçe)

Bu proje, otonom araç navigasyonunun temel bileşenlerinden biri olan şerit tespiti için OpenCV tabanlı bir bilgisayarlı görü (computer vision) boru hattı (pipeline) sunmaktadır. Sistem, yola kuşbakışı bir açıdan bakmak için perspektif dönüşümü uygular, HSV renk eşikleme ile şerit çizgilerini arka plandan ayırır ve histogram analizi ile şeritlerin başlangıç noktalarını belirler. Ardından, klasik Kayan Pencere (Sliding Window) algoritmasını kullanarak şeritleri kare kare takip eder ve hesaplanan güvenli sürüş rotasını orijinal video akışının üzerine yansıtır. Görüntü işleme tekniklerinin, matris dönüşümlerinin ve robotik alanındaki algoritmik düşünce yapısının pratik bir gösterimi olarak tasarlanmıştır.

# The Sliding Window Algorithm (Eng.)

The Sliding Window algorithm serves as the core tracking mechanism of this project. After applying the perspective transformation and color masking, the system computes a pixel histogram of the lower half of the image to identify the starting x-coordinates of the left and right lanes. From these base points, a series of rectangular "windows" sequentially slide upward across the image, isolating the non-zero pixels that represent the lane markings. By calculating the center of mass (mean position) for the pixels within each window, the algorithm dynamically adjusts the horizontal position of the subsequent window. This continuous recentering allows the system to accurately track and trace the lane lines, adapting to curves and changes in the road's trajectory.

# Kayan Pencere Algoritması (Türkçe)

Kayan Pencere algoritması, bu projenin temel şerit takip mekanizması olarak görev yapmaktadır. Perspektif dönüşümü ve renk maskeleme işlemlerinin ardından sistem, sol ve sağ şeritlerin başlangıç x-koordinatlarını tespit etmek için görüntünün alt yarısının piksel histogramını hesaplar. Bu temel noktalardan başlayarak, bir dizi dikdörtgen "pencere" görüntü boyunca sıralı olarak yukarı doğru kayar ve şerit çizgilerini temsil eden pikselleri izole eder. Her pencere içindeki piksellerin kütle merkezini (ortalama konumunu) hesaplayan algoritma, bir sonraki pencerenin yatay konumunu dinamik olarak günceller. Bu sürekli merkezleme işlemi, sistemin yolun kıvrımlarına ve yörünge değişikliklerine adapte olarak şerit çizgilerini yüksek hassasiyetle takip etmesini sağlar.



<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/bd844b19-2dfb-4605-93a2-f3a8fa63ee9a" />



# Adaptive Perspective Mapping (Dynamic Pitch Compensation)

To overcome the inherent limitations of fixed perspective transformations on inclined roads, this system addresses the geometric distortion caused by uphill and downhill gradients. Traditional static Region of Interest (ROI) selections fail when the vehicle's pitch changes, causing the bird's-eye view to misalign. To solve this, the pipeline can be enhanced with Adaptive Perspective Mapping. By dynamically adjusting the ROI source points based on real-time Vanishing Point Detection (or IMU sensor fusion), the algorithm ensures the transformation matrix remains perfectly parallel to the road surface. This adaptive approach prevents the sliding window logic from collapsing during elevation changes, ensuring robust lane tracking in highly dynamic, real-world driving conditions.

# Adaptif Perspektif Haritalama (Dinamik Eğim Telafisi)

Sabit perspektif dönüşümlerinin eğimli yollarda (yokuş aşağı/yukarı) yarattığı kaçınılmaz geometri bozulmalarının önüne geçmek için bu sistem, Adaptif Perspektif Haritalama yaklaşımını ele almaktadır. Geleneksel sabit İlgi Alanı (ROI) seçimleri, aracın yunuslama (pitch) açısı değiştiğinde kuşbakışı görünümün yoldan sapmasına neden olur. Bu sorunu çözmek için, ROI kaynak noktaları eşzamanlı Kaçış Noktası Tespiti (Vanishing Point Detection) veya IMU sensör füzyonu kullanılarak dinamik olarak güncellenmelidir. Bu sayede, dönüşüm matrisinin yol yüzeyiyle her zaman mükemmel şekilde hizalı kalması sağlanır. Bu adaptif yapı, yükseklik değişimlerinde kayan pencere (sliding window) algoritmasının çökmesini engeller ve oldukça dinamik olan gerçek dünya sürüş koşullarında stabil şerit takibi sunar.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/2ab3aec5-22a0-410a-bf9b-096148c0a2f4" />

# Probabilistic Hough Transform Lane Detection (Eng.)
This module implements a lightweight and highly efficient lane tracking pipeline optimized for real-time performance. It utilizes Canny Edge Detection to identify high-contrast structural outlines and applies a polygonal Region of Interest (ROI) mask to isolate the drivable path. The core detection engine leverages the Probabilistic Hough Line Transform (cv2.HoughLinesP) to mathematically extract linear lane segments from the edge map. These segments are dynamically separated into left and right boundaries based on their spatial slopes, averaged using 1st-degree polynomial fitting, and extrapolated to reconstruct the continuous lane lines. Designed with fault tolerance in mind, the system features a state-memory fallback mechanism to maintain a stable trajectory even when lane markings are temporarily obscured or missing.

# Olasılıksal Hough Dönüşümü ile Şerit Tespiti (Türkçe)

Bu modül, gerçek zamanlı performans için optimize edilmiş, sistem kaynaklarını yormayan yüksek verimli bir şerit takip boru hattı (pipeline) sunmaktadır. Yüksek kontrastlı yapısal hatları belirlemek için Canny Kenar Tespiti (Canny Edge Detection) algoritmasını kullanır ve sürülebilir rotayı izole etmek için çokgen bir İlgi Alanı (ROI) maskesi uygular. Temel tespit motoru, doğrusal şerit parçalarını kenar haritasından matematiksel olarak çıkarmak için Olasılıksal Hough Çizgi Dönüşümü'nden (cv2.HoughLinesP) yararlanır. Bu parçalar daha sonra uzamsal eğimlerine göre sol ve sağ sınırlar olarak ayrılır, 1. dereceden polinom uydurma (polynomial fitting) ile ortalamaları alınır ve kesintisiz şerit çizgilerini yeniden oluşturmak üzere uzatılır. Hata toleransı (fault tolerance) göz önünde bulundurularak tasarlanan sistem, şerit çizgilerinin geçici olarak kaybolduğu veya silikleştiği durumlarda yörüngeyi stabil tutmak için bir durum-bellek (state-memory) geri dönüş mekanizması barındırır.

<img width="1592" height="900" alt="image" src="https://github.com/user-attachments/assets/bdf97b78-bfcb-404d-bed1-ca771304995c" />
































