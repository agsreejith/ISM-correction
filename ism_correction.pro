; NAME:
;      ism_correction
;
; PURPOSE:
;      Outputs the difference in transit depth due to the effect of ISM.
;
; CALLING SEQUENCE:
;      ism_correction,Teff=teff,Radius=Radius,Rmg=Rmg,mgksigma=mgksigma,mghsigma=mghsigma,bb_td=bb_td,
;      mg_td=mg_td,ebv=ebv,n_mg2=n_mg2,vr_ISM=vr_ISM,nolog=0,tds_diff
;      All inputs are optional. The code has an interactive mode. to run in intractive mode call ism_correction
;      without keywords. 
; OPTIONAL INPUTS
;      
;      Teff     = Effective temperature of the star. 
;      Radius   = Radius of th e star. If not provided calculated based on effective temperature
;      Rmg      = Stellat chromspheric emission at 1 AU (in ergs/sec/cm2), if not specified the 
;                 code request for logR'HK
;      mgksigma = FWHM width of Mg II k line in A.
;      mghsigma = FWHM width of Mg II h line in A.           
;      bb_td    = Broad band transit depth in %
;      mg_td    = Transit depth at MgII lines in %.
;      p_width  = FWHM width of the planetary absorption feature in A.
;      ebv      = E(b-v) of the star. Used to calculate. THe code requires either E(b-v) or MgII
;                 column densities of ISM in the direction of observation.
;      n_mg2    = log MgII column density of ISM in the direction of observation. Required if E(b-v) 
;                 is not specified.
;      vr_ISM   = Relative radila velocity shift of the ISM feature. If not specifed assumes a 
;                 value of 0
;      nolog    = Set nolog to 1 to prevent creation of logfile.           
;      
; OUTPUT:
;      tds_diff = Difference in transit depth without and without ISM.
;
; PROCEDURE:
;      Calculates effect of ISM absorption at MgII lines on transit depth measurements;
;
;##################################################################################################

function find_nearest,array,value
  near = Min(Abs(array - value), index)
  return, index
end

function mg2_noism,flux_c,tds,dwl,fwhm,Mgaratio,MgII2w,sigmaMg22,MgII1w,sigmaMg21,E
  
  flux_noism = flux_c
  Mg21em=E/(1.+Mgaratio)
  Mg22em=Mgaratio*E/(1.+Mgaratio)
  gaussMg22=gaussian(flux_c[0,*],[0.3989*Mg22em/sigmaMg22,MgII2w,sigmaMg22])

  gaussMg21=gaussian(flux_c[0,*],[0.3989*Mg21em/sigmaMg21,MgII1w,sigmaMg21])

  gaussMg2 = gaussMg21 + gaussMg22

  flux_noism[1,*] = flux_c[1,*] + gaussMg2
  if dwl le (MgII2w-MgII1w) then begin
    mg2k_wst = MgII1w-(dwl/2.0d0)
    mg2k_wen = MgII1w+(dwl/2.0d0)
    mg2h_wst = MgII2w-(dwl/2.0d0)
    mg2h_wen = MgII2w+(dwl/2.0d0)
  endif else begin
    mg2_wst = ((MgII2w+MgII1w)/2.0)-(dwl)
    mg2_wen = ((MgII2w+MgII1w)/2.0)+(dwl)
  endelse
  ;stop
  flux_noism[1,*]=flux_noism[1,*]*tds[0,*]

  wave_noism=flux_c[0,*]
  spectra_noism=flux_noism[1,*]
  wave_new          = reform(wave_noism)
  photons_star_new  = reform(spectra_noism)
  hwhm              = fwhm/2
  ;convolution with instrument response
  smoothedflux_noism      = (gaussbroad(wave_new,photons_star_new,hwhm));/(4*!pi*(r_star^2))

  if dwl le (MgII2w-MgII1w) then begin
    mg2k_st = find_nearest(wave_new,mg2k_wst )
    mg2k_en = find_nearest(wave_new, mg2k_wen)
    mg2h_st = find_nearest(wave_new, mg2h_wst)
    mg2h_en = find_nearest(wave_new, mg2h_wen)
    mg2k_wave=wave_new[mg2k_st:mg2k_en]
    mg2k_flux=smoothedflux_noism[mg2k_st:mg2k_en]
    mg2h_wave=wave_new[mg2h_st:mg2h_en]
    mg2h_flux=smoothedflux_noism[mg2h_st:mg2h_en]
    mg2k_error=dblarr(n_elements(mg2k_wave))
    mg2k=trapz_error(mg2k_wave,mg2k_flux,mg2k_error)
    mg2h_error=dblarr(n_elements(mg2h_wave))
    mg2h=trapz_error(mg2h_wave,mg2h_flux,mg2h_error)
    mg2_noism=mg2k[0]+mg2h[0]
  endif else begin
    mg2_st = find_nearest(wave_new,mg2_wst )
    mg2_en = find_nearest(wave_new, mg2_wen)
    mg2_wave=wave_new[mg2_st:mg2_en]
    mg2_flux=smoothedflux_noism[mg2_st:mg2_en]
    mg2_error=dblarr(n_elements(mg2_wave))
    mg2hk=trapz_error(mg2_wave,mg2_flux,mg2_error)
    mg2_noism=mg2hk[0]
  endelse
  mg2noism_flux=mg2_noism;return,mg2_noism
  mg2noism={interflux:mg2noism_flux,flux:flux_noism}
  return,mg2noism 
end

function mg2_ism,flux_noism,dwl,MgII2w,MgII1w,fwhm,vr_ISM,n_mg2,ISM_b_Mg2,ismabs,ispl=ispl
  vc    = 299792.458d0      ;speed of light in km/s
  ;Parameters for MgII
  MgII1w      = 2795.5280 ;default value
  MgII1_loggf = 0.085     ;0.100 VALD/NIST
  MgII1_stark = -5.680
  MgII2w      = 2802.7050 ;default value
  MgII2_loggf = -0.218    ;-0.210 VALD/NIST
  MgII2_stark =-5.680     ;Stark damping constant
  sigmaMg21   = 0.22;0.288
  sigmaMg22   = 0.232;0.257
  ;ISM_b_Mg2   = 3         ;km/s
  fractionMg2 = 0.825     ;(Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
  Mg_abn      = -5.33     ;(Frisch & Slavin 2003; this is the ISM abundance of Mg)
  if dwl le (MgII2w-MgII1w) then begin
    mg2k_wst = MgII1w-(dwl/2.0d0)
    mg2k_wen = MgII1w+(dwl/2.0d0)
    mg2h_wst = MgII2w-(dwl/2.0d0)
    mg2h_wen = MgII2w+(dwl/2.0d0)
  endif else begin
    mg2_wst = ((MgII2w+MgII1w)/2.0)-(dwl)
    mg2_wen = ((MgII2w+MgII1w)/2.0)+(dwl)
  endelse
  ;n_flux=flux_noism[1,*]/flux_noism[2,*]
  ;for MgII doublet
  absorberMg1=create_struct('ion','MG21','N',n_mg2,'B',ISM_b_Mg2,'Z',0.0)
  lineMg1=create_struct('ion','Mg21','wave',MgII1w+MgII1w*vr_ISM/vc,'F',10^MgII1_loggf,'gamma',10^MgII1_stark)
  ISMMg21=voigtq(flux_noism[0,*],absorberMg1,lineMg1)

  absorberMg2=create_struct('ion','MG22','N',n_mg2,'B',ISM_b_Mg2,'Z',0.0)
  lineMg2=create_struct('ion','Mg22','wave',MgII2w+MgII2w*vr_ISM/vc,'F',10^MgII2_loggf,'gamma',10^MgII2_stark)
  ISMMg22=voigtq(flux_noism[0,*],absorberMg2,lineMg2)
  if ispl eq 0 then begin
    ISM = ISMMg21*ISMMg22
    flux_absorption = ISM ;* n_flux
    spectra=flux_noism[1,*]*flux_absorption
    ismabs =flux_noism[1,*]-spectra
  endif else begin
    spectra = flux_noism[1,*]-ismabs
    lt0 = where(spectra lt 0)
    spectra[lt0]=0.0
  endelse
  
  wave=flux_noism[0,*]
  ;spectra=flux[1,*]
  wave_new          = reform(wave)
  photons_star_new  = reform(spectra)
  hwhm              = fwhm/2
  smoothedflux      = (gaussbroad(wave_new,photons_star_new,hwhm));/(4*!pi*(r_star^2))

  if dwl le (MgII2w-MgII1w) then begin
    mg2k_st = find_nearest(wave_new,mg2k_wst )
    mg2k_en = find_nearest(wave_new, mg2k_wen)
    mg2h_st = find_nearest(wave_new, mg2h_wst)
    mg2h_en = find_nearest(wave_new, mg2h_wen)
    mg2k_wave=wave_new[mg2k_st:mg2k_en]
    mg2k_flux=smoothedflux[mg2k_st:mg2k_en]
    mg2h_wave=wave_new[mg2h_st:mg2h_en]
    mg2h_flux=smoothedflux[mg2h_st:mg2h_en]
    mg2k_error=dblarr(n_elements(mg2k_wave))
    mg2k=trapz_error(mg2k_wave,mg2k_flux,mg2k_error)
    mg2h_error=dblarr(n_elements(mg2h_wave))
    mg2h=trapz_error(mg2h_wave,mg2h_flux,mg2h_error)
    mg2=mg2k[0]+mg2h[0]
    ;tds[k]=1.0-(mg2[k]/mg2[0])
    ;mg2[j,k]=mg2k[0]+mg2h[0]
    ;tds[j,k]=1.0-(mg2[j,k]/mg2[j,0])
  endif else begin
    mg2_st = find_nearest(wave_new,mg2_wst )
    mg2_en = find_nearest(wave_new, mg2_wen)
    mg2_wave=wave_new[mg2_st:mg2_en]
    mg2_flux=smoothedflux[mg2_st:mg2_en]
    mg2_error=dblarr(n_elements(mg2_wave))
    mg2hk=trapz_error(mg2_wave,mg2_flux,mg2_error)
    mg2=mg2hk[0]
  endelse
  mg2ism={flux:mg2,ism:ismabs}
  return,mg2ism
end

pro ism_correction,Teff=teff,Radius=Radius,Rmg=Rmg,mghsigma=mghsigma,mgksigma=mgksigma,$
                   bb_td=bb_td,mg_td=mg_td,p_width=p_width,ebv=ebv,nmg2=nmg2,$
                   vr_ISM=vr_ISM,fwhm=fwhm,dwl=dwl,nolog=nolog,tds_diff


;===============================
;Constants
  R_sun = 6.957d10          ; cm
  AU    = 1.49598d13        ; cm (Astronomical Unit)
  vc    = 299792.458d0      ;speed of light in km/s
  c0    = 2.99792458d10     ; cm/s (speed of light)
  sigma = 5.67051d-5        ; erg/cm^2/s/K^4 (stefan-boltzmann)
  k_B   = 1.380658d-16      ; erg/K = g cm^2/s^2/K (boltzmann const)
  N_A   = 6.02214179d23     ; /mol (Avagadro constant)
  cA    = 2.99792458d18     ; Angstrom/s (speed of light)
;===============================
;Parameters for MgII

  MgII1w      = 2795.5280 ;default value
  MgII1_loggf = 0.085     ;0.100 VALD/NIST
  MgII1_stark = -5.680
  MgII2w      = 2802.7050 ;default value
  MgII2_loggf = -0.218    ;-0.210 VALD/NIST
  MgII2_stark =-5.680     ;Stark damping constant
  sigmaMg21   = 0.22;0.288
  sigmaMg22   = 0.232;0.257
  ISM_b_Mg2   = 3         ;km/s
  fractionMg2 = 0.825     ;(Frisch & Slavin 2003; this is the fraction of Mg in the ISM that is singly ionised)
  Mg_abn      = -5.33     ;(Frisch & Slavin 2003; this is the ISM abundance of Mg)
  Mgaratio_loggf2to1=(10^MgII2_loggf)/(10^MgII1_loggf)
  Mgratio=Mgaratio_loggf2to1
;===============================
;Parameter table
  T_book  = [46000,43000,41500,40000,39000,37300,36500,35000,34500,33000,32500,$
             32000,31500,29000,26000,24500,20600,18500,17000,16700,15700,14500,$
             14000,12500,10700,10400,9700,9200,8840,8550,8270,8080,8000,7800,$
             7500,7440,7220,7030,6810,6720,6640,6510,6340,6240,6170,6060,6000,$
             5920,5880,5770,5720,5680,5660,5590,5530,5490,5340,5280,5240,5170,$
             5140,5040,4990,4830,4700,4600,4540,4410,4330,4230,4190,4070,4000,$
             3940,3870,3800,3700,3650,3550,3500,3410,3250,3200,3100,3030,3000,$
             2850,2710,2650,2600,2500,2440,2400,2320]
  R_book  = [12.5,12.3,11.9,11.2,10.6,10,9.62,9.11,8.75,8.33,8.11,7.8,7.53,6.81,$
             5.72,4.89,3.85,3.45,3.48,3.41,3.4,2.95,2.99,2.91,2.28,2.26,2.09,2,$
             1.97,2.01,1.94,1.94,1.93,1.86,1.81,1.84,1.79,1.64,1.61,1.6,1.53,$
             1.46,1.36,1.3,1.25,1.23,1.18,1.12,1.12,1.01,1.01,0.986,0.982,0.939,$
             0.949,0.909,0.876,0.817,0.828,0.814,0.809,0.763,0.742,0.729,0.72,$
             0.726,0.737,0.698,0.672,0.661,0.656,0.654,0.587,0.552,0.559,0.535,$
             0.496,0.46,0.434,0.393,0.369,0.291,0.258,0.243,0.199,0.149,0.127,$
             0.129,0.118,0.112,0.111,0.107,0.095,0.104]
  SpT =   ['O3V','O4V','O5V','O5.5V','O6V','O6.5V','O7V','O7.5V','O8V','O8.5V',$
           'O9V','O9.5V','B0V','B0.5V','B1V','B1.5V','B2V','B2.5V','B3V','B4V',$
           'B5V','B6V','B7V','B8V','B9V','B9.5V','A0V','A1V','A2V','A3V','A4V',$
           'A5V','A6V','A7V','A8V','A9V','F0V','F1V','F2V','F3V','F4V','F5V',$
           'F6V','F7V','F8V','F9V','F9.5V','G0V','G1V','G2V','G3V','G4V','G5V',$
           'G6V','G7V','G8V','G9V','K0V','K0.5V','K1V','K1.5V','K2V','K2.5V',$
           'K3V','K3.5V','K4V','K4.5V','K5V','K5.5V','K6V','K6.5V','K7V','K8V',$
           'K9V','M0V','M0.5V','M1V','M1.5V','M2V','M2.5V','M3V','M3.5V','M4V',$
           'M4.5V','M5V','M5.5V','M6V','M6.5V','M7V','M7.5V','M8V','M8.5V','M9V',$
           'M9.5V']
;===============================
       
  ;Verfying keywords
  if N_params() lt 1 then begin
    print,'No input parameters present please specify the parameters as follows'
    print,'Stellar atmospheric parameters.'
    read,'Enter stellar Teff [K] : ',Teff
    if n_elements(Teff) eq 0 then begin
      read,' Please enter stellar Teff [K] : ',Teff
    endif
    if Teff lt 3495 or Teff gt 10500 then begin
      print,'Stellar temperature must be between 3500 and 10000.'
      read,'Please enter stellar Teff [K] in the above range: ',Teff
    endif
    if n_elements(Radius) eq 0 then begin
      read,'Enter stellar radius [Rsun] (if unknown enter -9.9): ',Radius
    endif
    if n_elements(Rmg) eq 0 then begin  
      read,'Enter the Mg line core emission (if unknown enter -9.9): ',Rmg
    endif
    if Rmg eq -9.9 then begin
      print,"Stellar activity (logR'HK needs to be entered)."
      read,"Enter logR'HK value (if unknown enter -9.9): ",logR
    
      if n_elements(logR) ne 0 then begin
        if logR lt -5.1 then begin
          logR=-5.1
          print,"Warning: the inserted logR'HK value is smaller than the basal level (-5.1)."+$
                " The logR'HK value is now set to be equal to the basal level."
        endif
      endif
    endif  
      if n_elements(logR) ne 0 then begin
       if logR lt -7. and Rmg eq -9.9 then begin
        print,"Please, enter at least logR'HK! or R_mg"
        stop
      endif
    endif
    if n_elements(mgksigma) eq 0 then begin
      read,' Please enter the width of  MgII k line (FWHM) [A] (if unknown enter -9.9) : ',mgksigma
    endif
    if n_elements(mghsigma) eq 0 then begin
      read,' Please enter the width of  MgII h line (FWHM) [A] (if unknown enter -9.9) : ',mghsigma
    endif
    print,''
    print,"Planetary parameters."
    read,'Enter the broad band transit depth [%]: ',bb_td
    read,'Enter the MgII transit depth [%]: ',mg_td
    read,'Enter the width of planetary absorption feature [A] (if unknown enter -9.9): ',p_width
    print,''
    print,"ISM parameters."
    read,'Enter the reddening E(B-V) [mag] (if unknown enter -9.9): ',ebv
    if float(ebv) ge 0. then begin
      nh=5.8d21*ebv
      nmg2        = alog10(nh*fractionMg2*10.^Mg_abn)
    endif
    if float(ebv) lt 0. then begin
      read,'Enter the MgII ISM column density [log scale]: ',nmg2
      read,'Enter the ISM radial velocity (relative to the stellar radial velocity) [km/s]'+$
        ' (if unknown enter 0.0): ',vr_ISM
    endif
    print,''
    print,"Instrument parameters."
    read,'Enter the resolution of the instrument [A]: ',fwhm
    read,'Enter the wavelength integration bin [A]: ',dwl
  endif
   ;===============================
   ;check for unavaliable keywords
   
  if (datatype(Teff) eq 'UND') then begin
    read,'Enter stellar Teff [K]: ',Teff
  endif  
  if (datatype(Radius) eq 'UND') then begin
    read,'Enter stellar radius [Rsun] (if unknown enter -9.9): ',Radius
  endif
  if (datatype(Rmg) eq 'UND') then begin
    print,"Stellar activity (logR'HK needs to be entered)."
    read,"Enter logR'HK value: ",logR
  endif
  if datatype(mgksigma) eq 'UND' then begin
    read,' Please enter the width of  MgII k line (FWHM) [A] (if unknown enter -9.9) : ',mgksigma
  endif
  if datatype(mghsigma) eq 'UND' then begin
    read,' Please enter the width of  MgII h line (FWHM) [A] (if unknown enter -9.9) : ',mghsigma
  endif  
  if (datatype(bb_td) eq 'UND') then begin
    read,'Enter the broad band transit depth [%]: ',bb_td
  endif
  if (datatype(mg_td) eq 'UND') then begin
    read,'Enter the MgII transit depth [%]: ',mg_td
  endif
  if datatype(p_width) eq 'UND' then begin
    read,'Enter the width of planetary absorption feature: ',p_width
  endif  
  if (datatype(ebv) eq 'UND') then begin
    if datatype(nmg2) eq 'UND' then begin
      read,'Enter the reddening E(B-V) [mag] (if unknown enter -9.9): ',ebv
      if float(ebv) ge 0. then begin
        nh=5.8d21*ebv
        nmg2        = alog10(nh*fractionMg2*10.^Mg_abn)
      endif
      if float(ebv) lt 0. then begin
        read,'Enter the MgII ISM column density [log scale]: ',nmg2
      endif
      if datatype(vr_ISM) eq 'UND' then begin
        read,'Enter the ISM radial velocity (relative to the stellar radial velocity)' +$
          '[km/s] (if unknown enter 0.0): ',vr_ISM
      endif
    endif
  endif
  if (datatype(dwl) eq 'UND') then begin
    read,'Enter the resolution of the instrument: ',fwhm
  endif
  if (datatype(fwhm) eq 'UND') then begin
    read,'Enter the wavelength integration bin: ',dwl
  endif
  if datatype(vr_ISM) eq 'UND' then vr_ISM=0.0
  if Radius le 0 then Radius=interpol(R_book,T_book,Teff)
  if p_width le 0.0 then p_width = 1.5
  if mgksigma gt 0 then sigmaMg21 = mgksigma*2*sqrt(2*alog(2))
  if mghsigma gt 0 then sigmaMg22 = mghsigma*2*sqrt(2*alog(2))
  dwl=dwl/2.0d0
  near = Min(Abs(T_book-Teff), index)
  stype     = SpT[index]
  ;BV     = interpol(BV_book,T_book,Teff)
  ;Load the stellar fluxes
  ;spawn,'gunzip fluxes.sav.gz'
  restore,'LLfluxes.sav'
  T=intarr(66)
  for i=0,n_elements(T)-1 do T[i]=3500.+100.*i
  mm=find_nearest(Teff,T) & mm=mm*2+1
  
  flux=dblarr(3,n_elements(ll[0,*]))
  flux[0,*]=ll[0,*]
  flux[1,*]=ll[mm,*]
  flux[2,*]=ll[mm+1,*]
  flux[1,*]=(flux[1,*]*cA)/(flux[0,*]^2.)
  flux[2,*]=(flux[2,*]*cA)/(flux[0,*]^2.)
  flux[1,*]=flux[1,*]*4.*!pi*(Radius*R_sun)^2*4*!pi ;convert to ergs/s/A second 4*!pi for steradian conversion
  flux[2,*]=flux[2,*]*4.*!pi*(Radius*R_sun)^2*4*!pi ;convert to ergs/s/A second 4*!pi for steradian conversion
  ;Rmg=0

  if (Rmg eq '-9.9') then begin
    if stype eq 'F0V' or'F1V' or stype eq 'F2V' or stype eq 'F3V' $
      or stype eq 'F4V' or stype eq 'F5V' or stype eq 'F6V' $
      or stype eq 'F7V' or stype eq 'F8V' or stype eq 'F9V' $
      or stype eq 'F9.5V' or stype eq 'G0V' or stype eq 'G1V' $
      or stype eq 'G2V' or stype eq 'G3V' or stype eq 'G4V' $
      or stype eq 'G5V' or stype eq 'G6V' or stype eq 'G7V' $
      or stype eq 'G8V' or stype eq 'G9V' then begin
      c1 = 0.87
      c2 = 5.73
      Rmg=10^(c1*logr+c2)
    endif  else if stype eq 'K9V' or stype eq 'K8V' or stype eq 'K7V' $
      or stype eq 'K6.5V' or stype eq 'K6V' or stype eq 'K5.5V' $
      or stype eq 'K5V' or stype eq 'K4.5V' or stype eq 'K4V' $
      or stype eq 'K3.5V' or stype eq 'K3V' or stype eq 'K2.5V'$
      or stype eq 'K2V' or stype eq 'K1.5V' $
      or stype eq 'K1V' or stype eq 'K0V' then begin
      c1 = 1.01
      c2 = 6.00
      Rmg=10^(c1*logr+c2)
    endif else if stype eq 'M9.5V' or stype eq 'M9V' or stype eq 'M8.5V' $
      or stype eq 'M8V' or stype eq 'M7.5V' or stype eq 'M7V' $
      or stype eq 'M6.5V' or stype eq 'M6V' or stype eq 'M5.5V' $
      or stype eq 'M5V' or stype eq 'M4.5V' or stype eq 'M4V' $
      or stype eq 'M3.5V' or stype eq 'M3V' or stype eq 'M2.5V'$
      or stype eq 'M2V' or stype eq 'M1.5V' $
      or stype eq 'M1V' or stype eq 'M0V' then begin
      c1 = 1.59
      c2 = 6.96
      Rmg=10^(c1*logr+c2)
    endif else begin
      Rmg=0
    endelse
    Rmg = Rmg*4*!pi 
  endif
  E=Rmg*AU^2
  WL=flux[0,*]
  Rp=dblarr(n_elements(WL))
  Rp=Rp+bb_td
  lFWHM = p_width ; full width height maximum
  sigma = lFWHM / (2. * SQRT(2.* ALOG(2.)))
  Rp_Mg1 = mg_td; peak in planetary radii
  Rp_Mg2 =Rp_Mg1*Mgaratio_loggf2to1
  ;Rp_Mg2 =Rp_Mg1*0.72609

  Rp_Mg21=Rp_Mg1-(Rp(where(WL eq MgII1w)))
  Rp_Mg22=Rp_Mg2-(Rp(where(WL eq MgII2w)))
  Mg21=(gaussian(WL, [Rp_Mg21,MgII1w,sigma]))
  Mg22=(gaussian(WL, [Rp_Mg22,MgII2w,sigma]))
  R = Rp+Mg21+Mg22;
  td = dblarr(3,n_elements(WL))
  td[1,*]=R
  mg2_noism_flux=dblarr(2)
  mg2_ism_flux=dblarr(2)
  tds_noism=dblarr(2)
  tds_ism=dblarr(2)
  ismabs=dblarr(n_elements(WL))
  mg2k_wst = MgII1w-(dwl/2.0d0)
  mg2k_wen = MgII1w+(dwl/2.0d0)
  mg2h_wst = MgII2w-(dwl/2.0d0)
  mg2h_wen = MgII2w+(dwl/2.0d0)
  
  for k=0,1 do begin
    tdval=(1-(td[k,*]/100.0))
    mg2noism=mg2_noism(flux,tdval,dwl,fwhm,Mgratio,MgII2w,sigmaMg22,MgII1w,sigmaMg21,E)
    mg2_noism_val=mg2noism.interflux
    flux_return=mg2noism.flux
    mg2_noism_flux[k]=mg2_noism_val
    if k gt 0 then isplanet = 1 else isplanet = 0
    mg2ism=mg2_ism(flux_return,dwl,MgII2w,MgII1w,fwhm,vr_ISM,nmg2,ISM_b_Mg2,ismabs,ispl=isplanet)
    mg2_ism_flux[k]=mg2ism.flux
    ismabs = mg2ism.ism
    tds_noism[k]=1.-(mg2_noism_flux[k]/mg2_noism_flux[0])
    if (mg2_ism_flux[k] gt 0 and mg2_ism_flux[0] gt 0) then begin
      tds_ism[k]=1.-(mg2_ism_flux[k]/mg2_ism_flux[0])
    endif else begin
      tds_ism[k] = 0.0
    endelse
    
  endfor
  tds_diff=tds_ism[1]-tds_noism[1]
  print,'Difference in transit depth [%]',tds_diff*100
  if datatype(nolog) eq 'UND' then nolog=0
  if nolog ne 1 then begin
    logfile='ism_correction_log_'+strtrim(string(Teff),2)+'.txt'
    loglun = 42
    openw, loglun, logfile
    printf,loglun,'Mg II ISM correction log'
    printf,loglun,'Input Parameters'
    printf,loglun,'Stellar Parameters'
    printf,loglun,'Stellar Temperature [K]       : ',Teff
    printf,loglun,'Stellar radius [R_sun]        : ',Radius
    printf,loglun,'Spectral type                 : ',Stype
    if (datatype(logR) ne 'UND') then begin
      printf,loglun,'Stellar activity index    : ',logR
    endif
    printf,loglun,'Mg II line emission @ 1AU     : ',Rmg
    printf,loglun,'Planetary Parameters'
    printf,loglun,'Broad band transit depth [%]  : ',bb_td
    printf,loglun,'Peak MgII k transit depth [%] : ',mg_td
    printf,loglun,'Planetary absorption width [A]: ',p_width
    printf,loglun,'ISM parameters'
    if (datatype(ebv) ne 'UND') then begin
      printf,loglun,'reddening E(B-V) [mag]    : ',ebv
    endif
    printf,loglun,'MgII column density           : ',nmg2
    printf,loglun,'ISM b-parameter [km/s]        : ',ISM_b_Mg2
    printf,loglun,'ISM radial velocity [km/s]    : ',vr_ISM
    printf,loglun,'Instrument paramters'
    printf,loglun,'Resolution [A]                : ',fwhm 
    printf,loglun,'Integration bin [A]           : ',dwl*2.0
    printf,loglun,'Output'
    printf,loglun,'Difference in transit depth [%]: ',tds_diff*100
    close,loglun
  endif  
end