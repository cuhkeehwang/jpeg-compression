import Image
import tkFileDialog
import time
from numpy import *


class Compression(object):
	def __init__(self, imagePath):
		
		self.paddedRow = 0
		self.paddedCol = 0
		self.originalImage = Image.open(imagePath)
		self.width, self.height = self.originalImage.size
		self.paddedImage = Image.new("RGB", (self.originalImage.size))
		self.yuvImage = Image.new("RGB", (self.originalImage.size))
		self.cssImage = Image.new("RGB", (self.originalImage.size))
		self.dctImage = Image.new("RGB", (self.originalImage.size))
		self.qImage = Image.new("RGB", (self.originalImage.size))
		self.dqImage = Image.new("RGB", (self.originalImage.size))
		self.idctImage = Image.new("RGB", (self.originalImage.size))
		self.finalImage = Image.new("RGB", (self.originalImage.size))
		
		self.yuvBlocks = self.splitImage(self.yuvImage,8)
		self.dctBlocks = self.splitImage(self.dctImage, 8)
		self.qBlocks = self.splitImage(self.qImage,8)
		self.dqBlocks = self.splitImage(self.dqImage,8)
		self.idctBlocks = self.splitImage(self.idctImage, 8)
		self.fBlocks = self.splitImage(self.finalImage,8)
		

		
		self.YUV = [
		[0.299, 0.587, 0.114],
		[-0.14713, -0.28886, 0.436],
		[0.615, -0.51499, -0.10001]
		]

		self.RGB = [
		[1,0,1.13983],
		[1,-0.39465,-0.58060],
		[1,2.03211,0]
		]
		#from page 256
		self.qLuminance = [
		[16,11,10,16,24,40,51,61],
		[12,12,14,19,26,58,60,55],
		[14,13,16,24,40,57,69,56],
		[14,17,22,29,51,87,80,62],
		[18,22,37,56,68,109,103,77],
		[24,35,55,64,81,104,113,92],
		[49,64,78,87,103,121,120,101],
		[72,92,95,98,112,100,103,99]
		]

		#from page 256
		self.qChrominance = [
		[17,18,24,47,99,99,99,99],
		[18,21,26,66,99,99,99,99],
		[24,26,56,99,99,99,99,99],
		[47,66,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99]
		]
	
		self.constantLow = [
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2],
		[2,2,2,2,2,2,2,2]
		]

		self.constantMedium = [
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32],
		[32,32,32,32,32,32,32,32]
		]
		
		self.constantHigh = [
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128]
		]
		
		self.nonUniformLow = [
		[8,5,5,8,12,20,25,30],
		[6,6,7,9,13,29,30,27],
		[7,6,8,12,20,28,34,18],
		[7,8,11,14,25,42,40,31],
		[9,11,18,28,35,54,51,38],
		[12,17,27,32,40,52,56,46],
		[24,32,39,43,51,60,60,50],
		[36,46,47,48,56,40,51,49]
		]

		self.nonUniformMedium = [
		[16,11,10,16,24,40,51,61],
		[12,12,14,19,26,58,60,55],
		[14,13,16,24,40,57,69,56],
		[14,17,22,29,51,87,80,62],
		[18,22,37,56,68,109,103,77],
		[24,35,55,64,81,104,113,92],
		[49,64,78,87,103,121,120,101],
		[72,92,95,98,112,100,103,99]
		]

		self.nonUniformHigh = [
		[64,44,40,64,96,160,204,244],
		[48,48,56,76,104,232,240,223],
		[56,52,64,96,160,228,276,224],
		[45,68,88,116,204,300,300,248],
		[73,88,148,224,272,300,300,300],
		[96,140,220,256,300,300,300,300],
		[196,256,300,300,300,300,300,300],
		[300,300,300,300,300,300,300,300]
		]
		
		self.sub = [
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128],
		[128,128,128,128,128,128,128,128]
		]
		
	def show(self, matrix):
	    # Print out matrix
	    for col in matrix:
	        return col

	def DCT_1D_Row(self, matrix):
		#F(u,v) - Horizontal 1D DCT
		S = self.createMatrix(8,8)
		for u in range(8):
			for v in range(8):
				temp = 0
				for i in range(8):
					temp +=cos((2*i+1)*u*pi/16) * matrix[i][v]
				if(u==0):
					c = sqrt(2)/2
				else:
					c = 1
				temp = temp*c/2
				S[u][v] = round(temp)
		return S

	def DCT_1D_Col(self,matrix):
		#G(i,v) - 1D DCT on columns
		S = self.createMatrix(8,8)
		for i in range(8):
			for v in range(8):
				temp = 0
				for j in range(8):
					temp += cos((2*j+1)*v*pi/16) * matrix[i][j]
				if(v==0):
					c = sqrt(2)/2
				else:
					c = 1
				temp = temp*c/2
				S[i][v] = temp
		return S

	def IDCT_1D_Row(self,matrix):
		S = self.createMatrix(8,8)
		for row in range(8):
			for i in range(8):
				temp = 0
				for u in range(8):
					if(u==0):
						c = sqrt(2)/2
					else:
						c = 1
					temp +=c*cos((2*i+1)*u*pi/16)*matrix[row][u]/2
				S[row][i] = round(temp) 
		return S

	def IDCT_1D_Col(self,matrix):
		S = self.createMatrix(8,8)
		for col in range(8):
			for i in range(8):
				temp = 0
				for u in range(8):
					if(u==0):
						c = sqrt(2)/2
					else:
						c = 1
					temp +=c*cos((2*i+1)*u*pi/16)*matrix[u][col]/2
				S[i][col] = round(temp) + 128
		return S

	#quantizes a matrix - done & tested
	def quantization(self, DCT, qMatrix):
		S = self.createMatrix(8,8)
		for row in range (8):
			for col in range (8):
				temp = round(float(DCT[row][col])/float(qMatrix[row][col]))
				S[row][col]=temp
		return S

	def dequantization(self,quantized, qMatrix):
		#dequantizes DCT coefficients -- tested and working
		S = self.createMatrix(8,8)
		for row in range (8):
			for col in range (8):
				temp = (quantized[row][col])*(qMatrix[row][col])
				S[row][col]=temp
		return S

	def getError(self,original, compressed):
			E = subtract(original,compressed)
			return E

	def convertColorSpace(self,im, colorspace):
		#converts image from RGB to YUV
		newIm = Image.new('RGB', im.size)
		putpixel = newIm.putpixel
		pixel = im.load()

		rPixel, gPixel, bPixel = self.getColorChannels(im)
		Y = self.createMatrix(self.width, self.height)
		for col in range(self.height):
			for row in range(self.width):
				Y_temp = self.colorspace[0][0]*rPixel[row,col] + self.colorspace[0][1]*gPixel[row,col] + self.colorspace[0][2]*bPixel[row,col]
				U_temp = self.colorspace[1][0]*rPixel[row,col] + self.colorspace[1][1]*gPixel[row,col] + self.colorspace[1][2]*bPixel[row,col]
				V_temp = self.colorspace[2][0]*rPixel[row,col] + self.colorspace[2][1]*gPixel[row,col] + self.colorspace[2][2]*bPixel[row,col]
				Y[row][col] = round(Y_temp)
				putpixel((row,col), (int(Y_temp), int(U_temp), int(V_temp)))
		return newIm

	def createMatrix(self,m,n):
		# Create zero m*n matrix
		matrix = [[0 for row in range(n)] for col in range(m)]
		return matrix

	#splits the image up into 8 by 8 blocks


	def reconstructImage(self, blocks, blocksize, imWidth, imHeight):
		height, width = blocks[0].size

		rowBlock = 0
		colBlock = 0

		im = Image.new('RGB', (imWidth,imHeight))
		putpixel = im.putpixel

		for block in range(len(blocks)):
			pixel = blocks[block].load()
			## changed 320 to imWidth
			if rowBlock != 0 and (rowBlock)%(imHeight/blocksize) == 0:
				colBlock += 1
			for i in range(blocksize):
				for j in range(blocksize):
					putpixel(((blocksize*colBlock + i)%imWidth,(blocksize*rowBlock + j)%imHeight), (pixel[i,j]))
			rowBlock +=1

		return im

	def chromaSampling(self, image):
		#4:2:0 chromasampling on UV channel
		blockRow = 0
		blockCol = 0
		chromaSample = Image.new('RGB', image.size, (255,255,255))
		
		chromaPixel = chromaSample.putpixel

		putpixel = image.putpixel
		chromaBlocks = self.splitImage(image, 2)

		for i in range(len(chromaBlocks)):
			yPixel, uPixel, vPixel = self.getColorChannels(chromaBlocks[i])
			averageU = int(round(average([uPixel[0,0], uPixel[0,1]])))
			averageV = int(round(average([vPixel[0,0], vPixel[0,1]])))
			if(blockRow==self.height):
				blockRow = 0
				blockCol += 2
			chromaPixel((1*blockCol,1*blockRow), (yPixel[0,0], averageU, averageV))
			chromaPixel((1*blockCol,1*blockRow+1), (yPixel[0,1], averageU, averageV))
			chromaPixel((1*blockCol+1,1*blockRow), (yPixel[1,0], averageU, averageV))
			chromaPixel((1*blockCol+1,1*blockRow+1), (yPixel[1,1], averageU, averageV))
			blockRow += 2
		self.cssImage = chromaSample

	#pads an image to so the height and width is divisible by 2
	def padImage(self, image):
		padCol = 0
		padRow = 0
		if self.width%8 != 0:
			padCol = 8 - self.width%8
		if self.height%8 != 0:
			padRow = 8 - self.height%8
		paddedImage = Image.new('RGB', (self.width+padCol, self.height+padRow))
		newpixel = paddedImage.putpixel
		oldpixel = image.load()

		for i in range (self.width):
			for j in range(self.height):
				newpixel((i,j), oldpixel[i,j])

		#PAD EXTRA COLUMNS
		for i in range(padCol):
			for j in range(self.height):
				value = oldpixel[self.width-1,j]
				newpixel((i+self.width,j), value)
		return padRow, padCol, paddedImage

		#update pixels
		oldpixel = paddedImage.load()

		#PAD EXTRA ROWS
		for i in range(padRow):
			for j in range(self.width+padCol):
				value = oldpixel[j,self.height-1]
				newpixel((j,i+height), value)
		return paddedCol, paddedRow, paddedImage
		


	#splits the pixels of an 8x8 image into 3 arrays containing (r,g,b) values respectively
	def pixelToArray(self, image):
		pixel = image.load()
		y = self.createMatrix(8,8)
		u = self.createMatrix(8,8)
		v = self.createMatrix(8,8)
		for i in range(8):
			for j in range(8):
				y[i][j] = pixel[i,j][0]
				u[i][j] = pixel[i,j][1]
				v[i][j] = pixel[i,j][2]
		return y,u,v

	#updates an 8x8 block from 3 matrices
	def updateImage(self, image,yMatrix, uMatrix, vMatrix):
		pixel = image.load
		putpixel = image.putpixel
		for i in range(8):
			for j in range(8):
				putpixel((i,j), (int(round(yMatrix[i][j])), int(round(uMatrix[i][j])), int(round(vMatrix[i][j]))))



	def compression(self, image, qChroma, qLuma):
		
		self.paddedCol, self.paddedRow, paddedImage = self.padImage(image)

		self.width, self.height = paddedImage.size
		
		imgBlocks = self.splitImage(paddedImage,8)
		self.yuvBlocks = self.splitImage(paddedImage,8)
		self.dctBlocks = self.splitImage(paddedImage, 8)
		self.qBlocks = self.splitImage(paddedImage,8)
		self.dqBlocks = self.splitImage(paddedImage,8)
		self.idctBlocks = self.splitImage(paddedImage, 8)
		self.fBlocks = self.splitImage(paddedImage,8)
		print len(imgBlocks), len(self.yuvBlocks), len(self.fBlocks)
			
		for block in range(len(imgBlocks)):
			rMatrix, gMatrix, bMatrix = self.pixelToArray(imgBlocks[block])
			#convert to YUV
			yMatrix, uMatrix, vMatrix = self.colorSpace(rMatrix,gMatrix,bMatrix,self.YUV)

			yMatrix_sub = subtract(yMatrix, self.sub)
			uMatrix_sub = subtract(uMatrix, self.sub)
			vMatrix_sub = subtract(vMatrix, self.sub)

			#1D Vertical DCT
			yMatrix_DCT_1 = self.DCT_1D_Col(yMatrix_sub)
			uMatrix_DCT_1 = self.DCT_1D_Col(uMatrix_sub) 
			vMatrix_DCT_1 = self.DCT_1D_Col(vMatrix_sub) 

			#1D Horizontal DCT
			yMatrix_DCT_2 = self.DCT_1D_Row(yMatrix_DCT_1)
			uMatrix_DCT_2 = self.DCT_1D_Row(uMatrix_DCT_1) 
			vMatrix_DCT_2 = self.DCT_1D_Row(vMatrix_DCT_1)

			#Luma quantization
			yMatrix_Q = self.quantization(yMatrix_DCT_2, qLuma)
			#Chroma quantization
			uMatrix_Q = self.quantization(uMatrix_DCT_2, qChroma)
			vMatrix_Q = self.quantization(vMatrix_DCT_2, qChroma)


			#Luma dequantization
			yMatrix_DQ = self.dequantization(yMatrix_Q, qLuma)
			#Chroma dequantization
			uMatrix_DQ = self.dequantization(uMatrix_Q, qChroma)
			vMatrix_DQ = self.dequantization(vMatrix_Q, qChroma)


			#1D Row IDCT
			yMatrix_IDCT_1 = self.IDCT_1D_Row(yMatrix_DQ)
			uMatrix_IDCT_1 = self.IDCT_1D_Row(uMatrix_DQ)
			vMatrix_IDCT_1 = self.IDCT_1D_Row(vMatrix_DQ)

			#1D Row IDCT
			yMatrix_IDCT_2 = self.IDCT_1D_Col(yMatrix_IDCT_1)
			uMatrix_IDCT_2 = self.IDCT_1D_Col(uMatrix_IDCT_1)
			vMatrix_IDCT_2 = self.IDCT_1D_Col(vMatrix_IDCT_1)

			rfMatrix, gfMatrix, bfMatrix = self.colorSpace(yMatrix_IDCT_2, uMatrix_IDCT_2, vMatrix_IDCT_2, self.RGB)

			self.updateImage(self.yuvBlocks[block], yMatrix, uMatrix, vMatrix)
			self.updateImage(self.dctBlocks[block], yMatrix_DCT_2, uMatrix_DCT_2, vMatrix_DCT_2)
			self.updateImage(self.qBlocks[block], yMatrix_Q, uMatrix_Q, vMatrix_Q)
			self.updateImage(self.dqBlocks[block], yMatrix_DQ, uMatrix_DQ, vMatrix_DQ)
			self.updateImage(self.idctBlocks[block], yMatrix_IDCT_2, uMatrix_IDCT_2, vMatrix_IDCT_2)
			self.updateImage(self.fBlocks[block], rfMatrix, gfMatrix, bfMatrix)
			print "Rendering block: " + str(block+1) + " of " + str(len(imgBlocks)) + " " 
		

		
		self.yuvImage = self.reconstructImage(self.yuvBlocks,8, self.width, self.height)
		self.chromaSampling(paddedImage)
		self.dctImage = self.reconstructImage(self.dctBlocks,8, self.width, self.height)
		self.qImage = self.reconstructImage(self.qBlocks,8, self.width, self.height)
		self.dqImage = self.reconstructImage(self.dqBlocks,8, self.width, self.height)
		self.idctImage = self.reconstructImage(self.idctBlocks, 8, self.width, self.height)
		self.finalImage = self.reconstructImage(self.fBlocks,8,self.width,self.height)

			#'%.2f'%(100*float(block)/float(len(imgBlocks))) + "..." 

	def colorSpace(self, xPixel, yPixel, zPixel, rgb):
		#converts image from RGB to YUV
		Y = self.createMatrix(8,8)
		U = self.createMatrix(8,8)
		V = self.createMatrix(8,8)
		for col in range(8):
			for row in range(8):
				Y_temp = rgb[0][0]*xPixel[row][col] + rgb[0][1]*yPixel[row][col] + rgb[0][2]*zPixel[row][col]
				U_temp = rgb[1][0]*xPixel[row][col] + rgb[1][1]*yPixel[row][col] + rgb[1][2]*zPixel[row][col]
				V_temp = rgb[2][0]*xPixel[row][col] + rgb[2][1]*yPixel[row][col] + rgb[2][2]*zPixel[row][col]
				Y[row][col]=Y_temp
				U[row][col]=U_temp
				V[row][col]=V_temp
		return Y,U,V
		
	def splitImage(self, image, blocksize):
		col = int(ceil(self.width/blocksize))
		row = int(ceil(self.height/blocksize))
	#	print col, row
		blocks = []
		for x in range(col):
			for y in range(row):
				blocks.append(image.crop((blocksize*x,blocksize*y, blocksize+blocksize*x, blocksize+blocksize*y)))
		return blocks

	#splits the image into 3 pixel arrays
	def getColorChannels(self, image):
		x,y,z = image.split()
		xPixel = x.load()
		yPixel = y.load()
		zPixel = z.load()	
		return (xPixel, yPixel, zPixel)
	
	def saveImages(self):
		self.originalImage.save("OriginalRGB.jpg")
		self.yuvImage.save("YUV.jpg")
		self.cssImage.save("ChromaSampled.jpg")
		self.dctImage.save("DCT.jpg")
		self.qImage.save("Quantized.jpg")
		self.dqImage.save("DQuantized.jpg")
		self.idctImage.save("IDCT.jpg")
		self.finalImage.save("FinalRGB.jpg")


