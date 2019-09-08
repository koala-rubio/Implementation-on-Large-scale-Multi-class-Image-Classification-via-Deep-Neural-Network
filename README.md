# Implementation-on-Large-scale-Multi-class-Image-Classification-via-Deep-Neural-Network

Deep learning is a research hotspot in machine learning in recent years. Compared with traditional machine learning, there are many advantages while employing deep learning, such as autonomous learning ability, automatic extraction of essential features, non-linear mapping ability, etc. Therefore, it is a very valuable research orientation to build the bridge between deep learning with image classification system as well as use convolutional neural network for image classification.

This paper mainly focuses on practical applications bas deep neural network in image classification from the following three aspects: model design and analysis, network optimization and prototype system design.

Firstly, the basic principles of several classical deep neural networks and their characters are summarized. Then, four network models are designed and built on tensorflow platform: BP neural network, convolution neural network, deep neural network and deep convolution neural network. Mnist and Cifar-10 datasets are used to train above four models to study the impact of the structure of datasets and network models on final classification results. Meanwhile, in order to maximize classification accuracy of the model, the following aspects are optimized: learning rate, batch size, activation function, network structure and dropout selection. These methods above have achieved fantastic results, which is evaluated and proved by this paper. The results show that the neural network model designed in this paper can achieve a high recognition rate on the Mnist dataset and a good classification accuracy on the Cifar-10, and the better the classification performance with the deepening of the network model.

Finally, based on the neural network model designed above, a prototype system for handwritten digital image classification is implemented. The prototype system can visually display the whole process of image classification based on neural network, process control and result display.













<!--
 /* Font Definitions */
@font-face
	{font-family:宋体;
	mso-font-charset:134;
	mso-generic-font-family:auto;
	mso-font-pitch:variable;
	mso-font-signature:3 680460288 22 0 262145 0;}
@font-face
	{font-family:"Cambria Math";
	panose-1:2 4 5 3 5 4 6 3 2 4;
	mso-font-charset:0;
	mso-generic-font-family:roman;
	mso-font-pitch:variable;
	mso-font-signature:-536870145 1107305727 0 0 415 0;}
@font-face
	{font-family:"\@宋体";
	mso-font-charset:134;
	mso-generic-font-family:auto;
	mso-font-pitch:variable;
	mso-font-signature:3 680460288 22 0 262145 0;}
 /* Style Definitions */
p.MsoNormal, li.MsoNormal, div.MsoNormal
	{mso-style-unhide:no;
	mso-style-qformat:yes;
	mso-style-parent:"";
	margin:0cm;
	margin-bottom:.0001pt;
	mso-pagination:widow-orphan;
	font-size:12.0pt;
	font-family:"Times New Roman",serif;
	mso-fareast-font-family:宋体;}
.MsoChpDefault
	{mso-style-type:export-only;
	mso-default-props:yes;
	font-size:10.0pt;
	mso-ansi-font-size:10.0pt;
	mso-bidi-font-size:10.0pt;
	mso-ascii-font-family:"Times New Roman";
	mso-fareast-font-family:宋体;
	mso-hansi-font-family:"Times New Roman";
	mso-font-kerning:0pt;}
 /* Page Definitions */
@page
	{mso-page-border-surround-header:no;
	mso-page-border-surround-footer:no;}
@page WordSection1
	{size:612.0pt 792.0pt;
	margin:72.0pt 90.0pt 72.0pt 90.0pt;
	mso-header-margin:36.0pt;
	mso-footer-margin:36.0pt;
	mso-paper-source:0;}
div.WordSection1
	{page:WordSection1;}
-->





