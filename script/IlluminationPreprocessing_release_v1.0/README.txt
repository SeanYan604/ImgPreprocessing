The released code is for academyic use purpose only, and WITHOUT ANY
WARRANTY; For any commercial use, please contact HanHu.CAS@gmail.com.


The LTV method demostrated here used a Total Variation package provided by Prof. Wotao Yin (http://www.caam.rice.edu/~optimization/L1/2007/09/software_08.html, GNU General Public License)


***************
To get the details of the 13 methods, please refer to our paper:

Hu Han, Shiguang Shan, Xilin Chen, and Wen Gao. A Comparative Study on Illumination Preprocessing in Face Recognition. Pattern Recognition (P.R.), vol. 46, no. 6, pp. 1691-1699, Jun. 2013.

Please cite the above paper, if our code is used in your research.
***************

Keywords: Face illumination preprocessing, face illumination normalizatioin, lighting normalizaition


How to run?

1¡¢Before running the code, you should add preproc2.m, gauss.m and gaussianfilter.m from http://lear.inrialpes.fr/people/triggs/src/amfg07-demo-v1.tar.gz, and unzip them into the 'IlluminationPreprocessing' directory.
      
2¡¢Run 'IlluminationNormalization4OneImg.m' and you can get 13 different images produced by 13 vary methods of illumination preprocessing.

3¡¢The parameters of some methods should be optimized based on your face image size. We used parameters for 64*80 face images. We align the face images based on the two-eye centers.