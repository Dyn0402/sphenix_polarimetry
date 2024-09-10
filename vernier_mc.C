void vernier_mc()
// Calculate collision vertex distribution, trans. and long., based on 
// two beam densities (trans. and long.); and many other effects
{
  // 1st beam shift
  const float DX1 = 0.0; // mm
  const float DY1 = 0.750; // mm
  const float DAX1 = 0.0e-3; // angle (rad)
  const float DAY1 = 0.5e-3; // angle (rad)
  // Beam sizes
  const float SX1 = 0.22; // mm
  const float SY1 = 0.22; // mm
  const float SZ1 = 1300; // mm, for beta*=1.0m
  //  const float SZ1 = 1150; // mm, for beta*=1.2m
  //  const float SZ1 = 1050; // mm, for beta*=1.5m
  const float SX2 = 0.22; // mm
  const float SY2 = 0.22; // mm
  const float SZ2 = 1300; // mm
  // beta*
  //  const float BETA = 1000; // mm
    const float BETA = 850; // mm
  //  const float BETA = 3000; // mm
//  const float BETA = 999999999; // mm

  const float Zoff = 1000;      // mm, offset of the collision vertex

  const float zBBC = 10440;    // mm, BBC position
  const float szBBC = 30;     // mm, BBC resolution
  const float szBBCtrig = 50;  // mm, BBC resolution on trigger level
  const float effBBC = 2000;  // mm, BBC eff vs z parameterized as Gaus(0,effBBC)
  const float zcutBBCtrig = 350; // mm, Z vertex cut for trigger BBCLL1

  const float szZDC = 150;     // mm, ZDC resolution
  const float szZDCtrig = 300;     // mm, ZDC resolution on trigger level
  const float zcutZDCtrig = 1500; // mm, Z vertex cut for trigger ZDCW

  const float ZMAX = 4000; // mm
  const float XYMAX = 3;   // mm
  const int NstepZ = 4000; // N of steps when moving in time
  const int NN = 1000; // N of points to calculate 3-dim integral (luminosity)

  float x01, y01, z01;
  float x02, y02, z02;
  float ztmp, zbbc, effbbc, zzdc, effzdc;
  float zbbc_trig, zzdc_trig;
  float sx1, sy1, sx2, sy2;
  float xx, yy, zz;
  float xloc, yloc, zloc;
  float l1, l2, ll;
  float zstep = 2*ZMAX/NstepZ;

  static TRandom* gen = new TRandom();

  TH1F* hx = new TH1F("hx","hx",300,-XYMAX,XYMAX);
  TH1F* hy = new TH1F("hy","hy",300,-XYMAX,XYMAX);
  TH1F* hz = new TH1F("hz","hz",400,-ZMAX,ZMAX);
  TH1F* hxcut = new TH1F("hxcut","hx",300,-XYMAX,XYMAX);
  TH1F* hycut = new TH1F("hycut","hy",300,-XYMAX,XYMAX);

  TH1F* hzbbc = new TH1F("hzbbc","hzbbc",400,-ZMAX/10,ZMAX/10);
  TH1F* hzzdc = new TH1F("hzzdc","hzzdc",400,-ZMAX/10,ZMAX/10);
  TH1F* hzbbc_zcut = new TH1F("hzbbc_zcut","hzbbc_zcut",400,-ZMAX/10,ZMAX/10);
  TH1F* hzzdc_zcut = new TH1F("hzzdc_zcut","hzzdc_zcut",400,-ZMAX/10,ZMAX/10);

  printf("Beta* = %3.1f m\n",BETA/1000);
  printf("Beam1 width (mm): sx=%4.2f, sy=%4.2f, sz=%5.0f\n",SX1,SY1,SZ1);
  printf("Beam2 width (mm): sx=%4.2f, sy=%4.2f, sz=%5.0f\n",SX2,SY2,SZ2);
  printf("Beam1 shift: %5.2f mm  Beam angle: %4.2f mrad\n",DX1,DAX1*1000);
  //  printf("Beam2 shift: %e mm  Beam angle: %e rad\n",DX2,DAX2);

  printf("\n Number of events to simulate %d\n\n",NstepZ);

  float rinp = 0;
  float rzdc = 0;
  float rbbc = 0;

  for( int it=0; it<NstepZ; it++ ) { // Like move in time

    if( it%100==0 ) printf("Step = %d (%d)\n",it,NstepZ);

    // Bunch center position
    z01 = -ZMAX + it*zstep;
    x01 = z01 * DAX1 + DX1;
    y01 = z01 * DAY1 + DY1;
    z02 =  ZMAX - it*zstep;
    x02 = 0;
    y02 = 0;

    for( int in=0; in<NN; in++ ) {
      xx = -XYMAX + 2*XYMAX*gen->Rndm();
      yy = -XYMAX + 2*XYMAX*gen->Rndm();
      zz = -ZMAX  + 2*ZMAX*gen->Rndm();
      // Hour glass effect
      sx1 = SX1 * sqrt(1+zz*zz/BETA/BETA);
      sy1 = SY1 * sqrt(1+zz*zz/BETA/BETA);
      sx2 = SX2 * sqrt(1+zz*zz/BETA/BETA);
      sy2 = SY2 * sqrt(1+zz*zz/BETA/BETA);
      // Shift
      xloc = xx-x01;
      yloc = yy-y01;
      zloc = zz-z01;
      // Rotation on small angle
      xloc -= zloc*DAX1;
      yloc -= zloc*DAY1;
      l1 = TMath::Gaus(xloc,0,sx1,true) * TMath::Gaus(yloc,0,sy1,true) 
	* TMath::Gaus(zloc,0,SZ1);
	//	* GetDensity_Rob(0,zloc);
      // Shift
      xloc = xx-x02;
      yloc = yy-y02;
      zloc = zz-z02;
      l2 = TMath::Gaus(xloc,0,sx2,true) * TMath::Gaus(yloc,0,sy2,true) 
	* TMath::Gaus(zloc,0,SZ2);
	//	* GetDensity_Rob(1,zloc);
      ll = l1*l2;

      hx->Fill(xx,ll);
      hy->Fill(yy,ll);
      hz->Fill(zz,ll);

      ztmp = zz+Zoff; // Vertex shift

      if( fabs(ztmp) < zcutBBCtrig ) {
	hxcut->Fill(xx,ll);
	hycut->Fill(yy,ll);
      }

      if( fabs(ztmp) < zcutBBCtrig ) rinp += ll;

      // BBC
      if( fabs(zz)<zBBC ) zbbc = gen->Gaus(ztmp,szBBC);
      else                zbbc = gen->Gaus(zBBC*(ztmp)/fabs(ztmp),szBBC);
      effbbc = TMath::Gaus(ztmp,0,effBBC);
      hzbbc->Fill(zbbc/10,ll*effbbc);
      // BBC with online vertex cut
      if( fabs(zbbc) < zcutBBCtrig ) rbbc += ll*effbbc;
      zbbc_trig = gen->Gaus(ztmp,szBBCtrig);
      if( fabs(zbbc_trig) < zcutBBCtrig ) hzbbc_zcut->Fill(zbbc/10,ll*effbbc);

      // ZDC
      zzdc = gen->Gaus(ztmp,szZDC);
      effzdc = 1;
      hzzdc->Fill(zzdc/10,ll*effzdc);
      if( fabs(zzdc) < zcutBBCtrig ) rzdc += ll*effzdc;
      // ZDC with online vertex cut
      zzdc_trig = gen->Gaus(ztmp,szZDCtrig);
      if( fabs(zzdc_trig) < zcutZDCtrig ) hzzdc_zcut->Fill(zzdc/10,ll*effzdc);
    }

  } // for( int it=0

  rinp /= hz->Integral();
  rbbc /= hzbbc->Integral();
  float rzdcw = rzdc;
  rzdc  /= hzzdc->Integral();
  rzdcw /= hzzdc_zcut->Integral();

  TFile* f = new TFile("profile.root","RECREATE");
  hx->Write();
  hy->Write();
  hz->Write();
  hxcut->Write();
  hycut->Write();
  hzbbc->Write();
  hzzdc->Write();
  hzbbc_zcut->Write();
  hzzdc_zcut->Write();
  f->Close();

  printf("Integral: %f\n",hz->Integral());
  printf("Integral bbc (with vert cut +/-%3.0f cm): %f\n",zcutBBCtrig/10,hzbbc_zcut->Integral());
  printf("Integral zdc (wide): %f\n",hzzdc_zcut->Integral());
  printf("Fraction in vertex cut +/-%3.0f cm: Rinp=%f Rbbc=%f Rzdc=%f (%f)\n",zcutBBCtrig/10,rinp,rbbc,rzdcw,rzdc);
}

//float GetDensity_Rob(int ibeam, float zz)
//// From data obtained from Robert
//// Input data created by from_robert/wcm_rob.C
//// Hist maximum is in bin=100
//{
//  if( ibeam!=0 && ibeam!=1 ) {
//    printf("Wrong beam ID=%d\n",ibeam);
//    return -999999999;
//  }
//
//  static bool first = true;
//
//  static int nn;
//  static float zmin;
//  static float zmax;
//  static float bwidth;
//  static float a1[1000];
//  static float a2[1000];
//
//  if( first ) {
//
//    TFile* f = new TFile("from_robert/wcm_205866.root");
//    TH1F* hyl = (TH1F*)f->Get("hylf");
//    TH1F* hbl = (TH1F*)f->Get("hblf");
//
//    nn = hyl->GetNbinsX();
//    bwidth = hyl->GetBinWidth(1);
//    zmin = hyl->GetBinLowEdge(1);
//    zmin -= bwidth*100; // Maximum (or z=0) is in bin=100
//    zmin *= 1000.;   // m -> mm
//    bwidth *= 1000.; // m -> mm
//    zmax = zmin+bwidth*nn;
//
//    for( int i=0; i<nn; i++ ) {
//      a1[i] = hyl->GetBinContent(i+1);
//      a2[i] = hbl->GetBinContent(i+1);
//    }
//    f->Close();
//
//    first = false;
//  }
//
//  float *pa;
//  if( ibeam==0 ) { pa = a1; }
//  else           { pa = a2; }
//
//  if( zz<=zmin || zz>=zmax ) return 0;
//
//  int iz = (zz-zmin)/bwidth;
//  return pa[iz];
//}
//
//
//float GetDensity(int ibeam, float zz)
//{
//  if( ibeam!=0 && ibeam!=1 ) {
//    printf("Wrong beam ID=%d\n",ibeam);
//    return -999999999;
//  }
//
//  static bool first = true;
//  static int nn;
//  static float z1[1000];
//  static float a1[1000];
//  static float z2[1000];
//  static float a2[1000];
//  Double_t ztmp, atmp;
//  if( first ) {
//    TFile* f = new TFile("from_kawall/prof_blue.root");
//    TGraph* gr = (TGraph*)f->Get("Graph");
//    nn = gr->GetN();
//    for( int i=0; i<nn; i++ ) {
//      gr->GetPoint(i,ztmp,atmp);
//      a1[i] = atmp;
//    }
//    f->Close();
//
//    TFile* f = new TFile("from_kawall/prof_yell.root");
//    TGraph* gr = (TGraph*)f->Get("Graph");
//    if( nn<2 || nn!=gr->GetN() ) {
//      printf("Error in GetDensity() when reading graphs: %d %d\n",nn,gr->GetN());
//    }
//    for( int i=0; i<nn; i++ ) {
//      gr->GetPoint(i,ztmp,atmp);
//      a2[i] = atmp;
//    }
//    f->Close();
//
//    for( int i=0; i<nn; i++ ) {
//      z1[i] = (i-100)*0.25*300; // mm
//      z2[i] = z1[i];
//      if( a1[i]<0 ) a1[i]=0;
//      if( a2[i]<0 ) a2[i]=0;
//      printf("%d: z=%f amp=%f %f\n",i,z1[i],a1[i],a2[i]);
//    }
//
//    first = false;
//  }
//
//  float *pz, *pa;
//  if( ibeam==0 ) { pz = z1; pa = a1; }
//  else           { pz = z2; pa = a2; }
//
//  if( zz<=pz[0] || zz>=pz[nn-1] ) return 0;
//
//  int i0=0;
//  while( i0<nn && zz>pz[i0] ) i0++;
//  return pa[i0];
//}



void vertex()
// The effect of BBC eff vs Z and BBC (ZDC) limited resolution on the
// measurement of the fraction of events within some vertex cut 
{
  //  const float sBeamZ  = 60;  // cm, collision vertex distr. width 
  //  const float sBeamZ  = 49;  // cm, collision vertex distr. width 
  const float sBeamZ  = 51;  // cm, collision vertex distr. width 
  const float zcut = 35;     // cm, z-vertex cut
  const float sBBCeff = 150; // cm, BBC eff. vs z
  const float sBBCz = 2;     // cm, BBC vertex resolution
  const float sZDCz = 15;    // cm, ZDC vertex resolution

  //  const float betastar = 99999999;
  //  const float betastar = 100;
  const float betastar = 100;

  const int nn = 10000;

  static TRandom* gen = new TRandom();
  TH1F* hz = new TH1F("hz","hz",100,-200,200);
  TH1F* hbbc = new TH1F("hbbc","hbbc",100,-200,200);
  TH1F* hzdc = new TH1F("hzdc","hzdc",100,-200,200);
  TH1F* hz_zcut = new TH1F("hz_zcut","hz_zcut",100,-200,200);
  TH1F* hbbc_zcut = new TH1F("hbbc_zcut","hbbc_zcut",100,-200,200);
  TH1F* hzdc_zcut = new TH1F("hzdc_zcut","hzdc_zcut",100,-200,200);

  float zz, zBBC, zZDC, ww;
  float effBBC, effZDC;
  float s0 = 0;
  float s0bbc = 0;
  float s0zdc = 0;
  float scut = 0;
  float sbbc = 0;
  float szdc = 0;

  for( int i=0; i<nn; i++ ) {
    zz = gen->Gaus(0,sBeamZ);
    zBBC = gen->Gaus(zz,sBBCz);
    zZDC = gen->Gaus(zz,sZDCz);
    effBBC = TMath::Gaus(zz,0,sBBCeff);
    effZDC = 1;
    ww = (1.+zz*zz/betastar/betastar);

    s0 += 1*ww;
    s0bbc += effBBC*ww;
    s0zdc += effZDC*ww;

    hz->Fill(zz,1*ww);
    hbbc->Fill(zBBC,effBBC*ww);
    hzdc->Fill(zZDC,effZDC*ww);

    if( fabs(zz)<zcut )   {
      scut += 1*ww;
      hz_zcut->Fill(zz,1*ww);
    }
    if( fabs(zBBC)<zcut ) {
      sbbc += effBBC*ww;
      hbbc_zcut->Fill(zBBC,effBBC*ww);
    }
    if( fabs(zZDC)<zcut ) {
      szdc += effZDC*ww;
      hzdc_zcut->Fill(zZDC,effZDC*ww);
    }
  }

  printf("Zcut   : %f\n",scut/s0);
  printf("BBC cut: %f\n",sbbc/s0bbc);
  printf("ZDC cut: %f\n",szdc/s0zdc);

  // Save hists
  TFile* f = new TFile("vert.root","RECREATE");
  hz->Write();
  hbbc->Write();
  hzdc->Write();
  hz_zcut->Write();
  hbbc_zcut->Write();
  hzdc_zcut->Write();
}


void fit()
{
  const int nn=4;
  float d[nn] = { 0,    0.3,   0.6,   0.9};

  //  float l[nn] = { 21908,  13939,  3919,  404}; // BBC
  //  float l[nn] = { 50567,  34802, 12238, 2581}; // ZDC
  //  float l[nn] = { 67620,  40021,  9509, 1268}; // ZDC
  float l[nn] = { 4373,  2750,  686, 68};

  // b*=99999m: actual z
  //  float l[nn] = { 5454,  3425, 847,  82.5};
  // b*=1m: actual z
  //  float l[nn] = { 5250,  3353, 873,  93.1};

  // b*=99999m: BBC
  //  float l[nn] = { 5554,  3494, 863,  83.9};
  // b*=1m: BBC
  //  float l[nn] = { 5336,  3415, 892,  95.4};

  // b*=1m: BBC with vert cut
  //  float l[nn] = { 2196,  1404, 371.6, 39.7 };
  // b*=1m: ZDC
  //  float l[nn] = { 5217,  3648, 1336,  323 };

  // b*=1m: BBC with ONLINE vert cut (35cm)
  //  float l[nn] = { 2155,  1365, 360.6, 38.9 };
  // b*=1m: ZDC with ONLINE vert cut (150cm)
  //  float l[nn] = { 4991,  3438, 1209,  256 };

  // b*=1.2m: BBC with vert cut
  //  float l[nn] = { 1944,  1237, 321.9, 33.3 };
  // b*=1.2m: ZDC
  //  float l[nn] = { 4601,  3140, 1046, 202.3 };

  // b*=1.5m: BBC with vert cut
  //  float l[nn] = { 1775,  1126, 288.8, 29.1 };
  // b*=1.5m: ZDC
  //  float l[nn] = { 4207,  2805, 853.4, 132.3 };

  // b*=99999m, bunch length as for b*=1.5m: BBC with vert cut
  //  float l[nn] = { 1808,  1139, 284.3, 27.4 };

  // b*=1m: BBC with vertex cut
  //  float l[nn] = { 8714,  5618, 1476,  159};
  // b*=1m: ZDC
  //  float l[nn] = {20027, 13955, 5024,  1165};
  // b*=99999: BBC with vertex cut
  //  float l[nn] = { 9088,  5755, 1426,  139};
  // b*=99999: BBC with vertex cut
  //  float l[nn] = { 9088,  5755, 1426,  139};
  // b*=99999: BBC
  //  float l[nn] = { 26341, 16472, 4080,  400};
  // b*=1m: BBC
  //  float l[nn] = { 1286, 990, 483, 150};


  // With hour glass effect
  //  float d[nn] = { 0,    0.1,   0.2,   0.3};
  //  float l[nn] = {1120, 1069, 918.6, 710.0};
  // With hour glass effect, within +/-30cm
  //  float d[nn] = { 0,    0.1,   0.2,   0.3};
  //  float l[nn] = {561.7, 536.8, 454.1, 339.3};
  // With hour glass effect, within +/-10cm
  //  float d[nn] = { 0,    0.1,   0.2,   0.3};
  //  float l[nn] = {223.2, 210.6, 176.5.1, 131.2};

  //  const int nn2 = (nn-1)*2 + 1;
  const int nn2 = 7;
  float xx[nn2], yy[nn2];

  for( int i=0; i<nn; i++ ) {
    xx[i] = -d[nn-1-i];
    yy[i] =  l[nn-1-i];
    xx[i+nn-1] = d[i];
    yy[i+nn-1] = l[i];
  }

  TGraph* gr = new TGraph(nn2,xx,yy);
  gr->SetMarkerStyle(20);
  gr->Draw("AP");
  TF1* fun = new TF1("fun","gaus");
  gr->Fit(fun);
}
