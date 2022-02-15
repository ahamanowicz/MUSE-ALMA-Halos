import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const
from scipy.stats import skewnorm
from scipy.integrate import quad
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Ellipse
from scipy.signal import gaussian, fftconvolve
from astropy.io import fits
from astropy.wcs import WCS
import astropy
import scipy.integrate as integrate
from scipy.ndimage.interpolation import rotate
from astropy import units as u
########## colours - Hogwart is my Home palette #######
#Hogwart is my Home palette
red='#b70000'
yellow='#ffc600'
blue='#000f9e'
green='#0f8400'
red='#b70000'
yellow='#ffc600'
blue='#000f9e'
green='#0f8400'

#########################################################################################

### signal shape #####

## gaussian ##

def gauss(x, mu, fwhm,A):
	
	"""
	mu - peak frequency
	sig - width (FWHM) / [ 2 * sqrt(2*ln(2)) ]
	A - peak flux

	"""
	sig = fwhm/(2*np.sqrt(2*np.log(2)))
	return A*np.exp(-np.power((x - mu)/sig, 2.)/2)

def boxy(x,mu,wid,A):
	"""
	mu - central frequency
	wid -  width of full box ( borders mu - wid/2 ; mu + wid/2)
	A - peak flux
	"""
	box = np.array([])
	for f in x:
	
		if f < mu-wid/2 or f > mu+wid/2:
			box=np.append(box,0)
		elif f >= mu-wid/2 and f<= mu+wid/2:
			box=np.append(box,A)
	return box

def horn(x,mu,wid,A):
	"""
	mu - central frequency
	wid -  width of full box ( borders mu - wid/2 ; mu + wid/2)
	A - peak flux
	used x^4 polynomial
	"""
	po = 4
	a = A/3./(wid/2.)**po
	b= 1.*A/3.
	
	#abs(((x-mu))**2)*a+b

	box = np.array([])
	for f in x:
	
		if f < mu-wid/2 or f > mu+wid/2:
			box=np.append(box,0)
		elif f >= mu-wid/2 and f<= mu+wid/2:
			box=np.append(box,abs((f-mu)**po)*a+b)
	return box
 

##########################################################################################

############## upload the cube & read parameters ####################

#opne cube 
#opne cube 
filein='fits_list.rms'

cubes=np.loadtxt(filein, usecols=(0,1),dtype='str')
count=0

for c in cubes:	
        
        count+=1
        print str(count) + " / " + str(np.shape(cubes)[0]) 
	cubename=c[0]
	rms_cube=float(c[1])
	hdulist = fits.open(cubename)
	data=hdulist[0].data
	wcs = WCS(hdulist[0].header)
	if np.size(hdulist) == 2 :
	
		beams=hdulist[1].data
		### read the beam
		cell= astropy.wcs.utils.proj_plane_pixel_scales(wcs)*3600	#cell size
		beam_a = np.median(beams['BMAJ'])/cell[0]			 		#in pixel 2 x major axis
		beam_b = np.median(beams['BMIN'])/cell[1] 					#in pixel  2 x minor axis
		beam_pa = np.median(beams['BPA'])							#for plotting

	else:
		head = hdulist[0].header
		cell= astropy.wcs.utils.proj_plane_pixel_scales(wcs)#*3600	#cell size
		beam_a = head['BMAJ']/cell[0]			 		#in pixel 2 x major axis
		beam_b = head['BMIN']/cell[1] 					#in pixel  2 x minor axis
		beam_pa = head['BPA']							#for plotting
                print "header"
	####### cube parameters #######


	### XYZ size ###

	# edges of the cube - from cube formation X = Y, imsize is always defiend as the square
	x1 = y1 = 0
	x2 = np.shape(data)[3]	
	y2 = np.shape(data)[2]																	
	pix =  np.shape(data)[3]						# nuber of pixels in one axis (same for X and Y in this cube formation pattern )
	X = Y = range(x1,x2)							# number of pixels in each direction
	chn = np.shape(data)[1]							# number of channels , Z

	Z = range(1,chn)								# frequency slices - channels



	### frequency coverage ###

	freq_1 = hdulist[0].header['CRVAL3'] 			# Hz | edge of the cube in frequency
	df = hdulist[0].header['CDELT3']		
	freq_2 =(freq_1 + (chn-1)*df) 					# Hz | edge of the cube in frequency

	freq_1 = freq_1/1.e9							# convert to GHz
	freq_2 = freq_2/1.e9							# convert to GHz

	if freq_2 > freq_1:
	 
		freq = np.linspace(freq_1,freq_2,chn) 		# GHz 

	else: 

		freq = np.linspace(freq_2,freq_1,chn) 		# GHz 

	chan_width = abs(freq_1 - freq_2)/chn 			# width of a channel in GHz



	#print "chan_width: "+str(chan_width) + " GHz"
	#print "freq ["+str(freq_1)+" ; "+str(freq_2)+ "] GHz"
		
	### noise parameters ###

	sigma_n = rms_cube								# sigma_noise defied as th erms of the whole cube [Duchamp way]

	#print "noise sigma: "+str(round(sigma_n*1e3,3))+ " mJy/beam"

	g = open(cubename.replace(".fits",".stats"), "w")
	g.write("cubename"+'\t'+ cubename+"\n")
	g.write("rms"+'\t'+ str(rms_cube)+"\n")
	g.write("cell [arcsec, arcsec]"+'\t'+ str(cell[0:2])+"\n")
	g.write("beam_axes [pix, pix]"+'\t'+ str(beam_a)+" "+str(beam_b) +"\n")
	g.write("beam_PA [deg]"+'\t'+ str(beam_pa) +"\n")
	g.write("image_size [pix pix]"+'\t'+ str(x2)+" "+str(y2) +"\n")
	g.write("channels"+'\t'+ str(chn) +"\n")
	g.write("frequency_range [GHz]"+'\t'+ '\t'+ str(freq_1)+" "+str(freq_2) +"\n")
	g.write("channel_width [GHz]"+'\t'+ str(chan_width) +"\n")
	g.close()
	################ signal parameters ##################

	n_mock = 20 # how many mock signals plugged

	#detection array

	### create the list of mock signals 

	Xs,Ys,Zs,Freqs,FPs,SNs,Ws,Shs=[],[],[],[],[],[],[],[] # saving moc signal parameters

	hdulist.close()
	
	f = open(cubename.replace('.fits','_mock.txt'), 'w')
	f.write("OBJ_ID" + '\t' + "X" + '\t' + "Y" + '\t' + "Z" + '\t' + "Frequency" + '\t' + "F_peak" + '\t' + "S/N" + '\t' + " width" + '\t' +'\t' + "n_chann" + '\t'+"shape" + '\t'+ "F_integrated" + '\n')	
	f.write("[]" + '\t' + "[pix]" + '\t' + "[pix]" + '\t' + "[pix]" + '\t' + "[Hz]" + '\t'+'\t' + "[mJy/beam]"  + '\t' +'[]'+'\t'+ " [km/s]"  + '\t'+"[]" +'\t'+"[]" + '\t'+ "[mJy/beam]" + '\n')	


	Xposition = range(x1+3,x2-3)				# XY position space for random (not centralized on any pixel)
	Yposition = range(y1+3,y2-3)				# XY position space for random (not centralized on any pixel)
	#print str(m) + " / "  + str(M)				# 3 pix frame included - ommitting the region
	Xs,Ys,Zs,Freqs,FPs,SNs,Ws,Shs=[],[],[],[],[],[],[],[] # saving moc signal parameters
	detection= np.zeros(np.shape(data))				# detection array - same size as the cube

	for N in range(n_mock):

		#print str(N) + " / 20"
		### position XYZ ###

		## mean freq ##

		freq_space = np.linspace(freq_1,freq_2,1000) 	# freq space for random
		f0 = np.random.choice(freq_space) 				# central frequency
		Z0 = 0 											# Z corresponding to mean frequency of the signal (find below)
		#print "mean freq: "+str(f0)+ " GHz" 

		for i in range(chn):
			if abs(f0 - freq[i]) < chan_width/2.:
				Z0=i
		#print "Z0: "+str(Z0) 							# central frequency slice

		### F_peak ###

		sig = np.linspace(1,8,1000)						# sigma space for random
		A = np.random.choice(sig)						# choose sigma level of signal
		F_peak = A*sigma_n 								# Jy/beam 
			
		#print "F peak = "+str(round(F_peak,3))+" S/N = " + str(round(A,3))

		### width and integrated flux (ideal) ###

		
		#km/s # width [km/s] to width [GHz]   
		#df/f0 = dv/c => df = dv/c * f0
		#f_width = width/const.c * f0 [GHz]

		width = np.linspace(50,800,10000) 				# [km/s] velocity space for random			
		width0 = np.random.choice(width)				# choose the width [km/s]
		fwidth0 = width0*1.e3/const.c *f0 				# conver [km/s] to [GHz]

		n_chan = fwidth0/chan_width 						#widdth of detection in channels

		#print "width = "+str(round(width0,3))+" km/s"

		## shape of signal ##


		#0 - gaussian
		#1 - boxy
		#2 - 2-horn
		

		shape = [0,1,2]									# shape space for random
		shape0 = np.random.choice(shape)				# choose the signal shape

		if shape0 == 0:
			signal = gauss(freq,f0,fwidth0,F_peak)		# model 
			F_int,err = integrate.quad(gauss, freq[0], freq[-1],args=(f0,fwidth0,F_peak))
			#print "shape: gaussian"
		elif shape0 == 1:
			signal = boxy(freq,f0,fwidth0,F_peak)		# model
			F_int = np.trapz(signal,freq)
			#print "shape: box"
		else:
			signal = horn(freq,f0,fwidth0,F_peak)		# model 
			F_int = np.trapz(signal,freq)
			#print "shape: double horn"


		### XY position ###
		#choosing the pixel representing unresolved source

		X0=np.random.choice(Xposition)						# choose x0 					
		Y0=np.random.choice(Yposition)						# choose y0
		#print X0
			#check if the position isn't already taken +/- 10 pix around
	
		if N > 0:
			for n in range(N):
				X1=int(Xs[n])
				Y1 =int(Ys[n])
				if np.linalg.norm(np.array([X1,Y1])-np.array([X0,Y0])) < 10:
					while np.linalg.norm(np.array([X1,Y1])-np.array([X0,Y0])) < 10:
				
						X0=np.random.choice(Xposition)						# choose x0 					
						Y0=np.random.choice(Yposition)	
			
		#plug-in detection - delta function

		#####. delta funcion representing unresolved detection #####
		for z in Z:
			detection[0][z][Y0][X0] = signal[z]


		Xs.append(str(X0))
		Ys.append(str(Y0))
		Zs.append(str(Z0))
		Freqs.append(str(f0*1e9))
		FPs.append(str(round(F_peak*1e3,5)))
		SNs.append(str(round(A,3)))
		Ws.append(str(round(width0,3)))
		Shs.append(str(shape0))
		
		f.write(str(N+1) + '\t' + str(X0) + '\t' + str(Y0) + '\t' + str(Z0) + '\t' + str(f0*1e9) + '\t' + str(round(F_peak*1e3,5)) + '\t' + str(round(A,3)) + '\t'+ str(round(width0,3)) + '\t'+ '\t' + str(int(n_chan))+'\t'  + str(shape0) +'\t'+str(round(F_int*1e3,3))+'\n')

	### beam convolution ####

        kernel = np.outer(gaussian(x2, beam_a/2.35), gaussian(y2, beam_b/2.35))		#kernel FWHM to sigma
        kernel = rotate(kernel,beam_pa/2.) #rotation of the beam according to the PA
        data_new = data*0

        convolved =  np.zeros(np.shape(data))
        for z in Z:


                Slice = detection[0][z]
                blurred = fftconvolve( Slice,kernel, mode='same')
                convolved[0][z]=blurred

        data_new = data + convolved


        f.close()

        ### save fits file #####
        hdu=fits.PrimaryHDU(data_new, header=hdulist[0].header)
        if np.size(hdulist) == 2:
                hdu2=fits.TableHDU(beams, header=hdulist[1].header)
        hdul = fits.HDUList([hdu])
        hdu.writeto(cubename.replace('.fits','_mock.fits'))
        hdulist.close()

