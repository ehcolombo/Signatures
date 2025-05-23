#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_statistics.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_fft_complex.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <omp.h>


double dt_0=1.0/(24); //minutes in units of days
double dt=dt_0;
//// 
double deltaT=0;
double acc_dt=0;
double T=0;
int L=100; // cellsxcelss
double LCUTOFF=1; // km = 10^5 cm
double ode_epsabs=0.001;
double Lparm=0;
float dx = 1.0;

double BETA,a1,b1,a2,b2;

double TAU=25;
double ALPHA=6.5;


//float timescale=1.0/(3600*24); // from days to second
float timescale=1.0; // in days
double total_time=30;//24*3600;//3600*24*12; // 30 days
float lengthscale=1.0;
double saveData_time_1 = 1.0/24-dt;//every hour
double saveData_time_2 = 0.25-dt;//every half-day
//
double updateRates_time = 0;
double sub_time_1=0;
double sub_time_2=0;
int count_save=0;
//
float D=0.000;
double psiD=0;
//
float p_reproduction_rate=timescale*1.0;
float p_death_rate=0;
float p_carrying_capacity=100;
float z_reproduction_rate=0.01;
float z_catching_rate=timescale; //30 (ingestion every 10s at high p leads to c = 864)
float z_bodysize=0.10/lengthscale; // cm estimated spherical radius
float z_catching_range=2*z_bodysize/lengthscale; //catching range/radius
float z_perceptual_range=10*z_bodysize/lengthscale; //0.5 cm body size - zooplankton
float z_death_rate=timescale*0.1;
float p_perceptual_range=0.01; //0.0005 cm body size
float p_bodysize=0.01; //0.0005 cm body size
float z_b0=10000000; //par of the response function of the catching rate
//double zoo_jumprate=1.0;//10*timescale*(24*3600);
double taur=8;
double gammar=1.0/taur;//1.0/5;

double timescale_factor = float(24*3600)/(100*1000); // 1 km/day = factor * cm/s

double rateTOT=0;
double birthTOT=0;
double deathTOT=0;
double pbirthTOT=0;
double cRateTOT=0;

double total_Z=0;
double total_P=0;
double total_N=0;

double nextT_bd=0;
double DFT_p0=0;
double DFT_fx=0;


double velocityX_eddy=0;
double velocityY_eddy=0;

double avgV=0;
double maxV=0;
double avgDistPP=0;
double avgDistZZ=0;
double avgDistZP=0;
int n_phyto_mf;
int n_zoo_mf;
double eff_c=1.0;
int Z_toggle=0;
float z_cost_0=1;
float z_benefits_0=10;
int strat=0;


int n_particle=1000;
int n_phyto = 0;
int n_zoo = 0;
int n_max_particles=100000;
int n_max_ng=100;

double CHI=0.1;
double CHIF=0;
double zoo_V0 = 1.0;

double GAMMAexp=0.25;

double P_stickiness = 0.15;


double z_ingestionRate=0;
double z_growthRate=0;
double z_experienceP=0;
double ClumpingZ=0;
double ClumpingP=0;
double CatchingZ=0;
double EncounterZP=0;
double EncounterChiZP=0;
double EncounterTargetZP=0;
double experienceZ=0;
double shareP=0;
double ChiZ=0;
int countINFO=0;

int trial=0;
int n_trial=1;

// double a_eddy=0.14;//*10;//*(3600*24)/timescale;
// int n_eddies=1400;//2000;
// double R_max=(float)64;
// double R_min=(float)1;//0.0001 cm, 0.001 m, 1 km
// double pexp=4.333;//4.3333;
// int resolution=int(N);


int n_eddies=100;
double a_eddy=1.0;//*(3600*24)/timescale;
double a_eddy_Real=a_eddy;
double R_max=(float)L/4;
double R_min=(float)L/4;//0.0001 cm, 0.001 m, 1 km
double pexp=4.333;//4.3333;
int N=L*L;
int resolution=N;

double meanCatching=0;
double meanClumping=0;
double meanCatchingTarget=0;
double meanEncounterZP=0;
double meanEncounterChiZP=0;
double meanEncounterTargetZP=0;
double meanExperienceZ=0;
double meanShareP=0;
double meanCount=0;


gsl_rng * r;
int seed;  

FILE * fdistribution;
FILE * fspectra;
FILE * fevolution;
FILE * fvector;
FILE * fhistogram;

char file1_name[99];
char file2_name[99];
char file3_name[99];
char file4_name[99];
char file_info[99];

double *vecField_ft_avg;
fftw_complex *flowSpectrum;
double ***vectorGrid;
double *flowDeltaVelocities;

double *Su;
double *Sv;
double *Sw;
double *Sf;

double *y_eddies = NULL;
gsl_odeiv2_system sys_eddy;
gsl_odeiv2_driver * d_eddy;
gsl_odeiv2_system sys_dens;
gsl_odeiv2_driver * d_dens;


struct eddy{
	double x;
	double y;
	double w;
	double R;
	double vx;
	double vy;
};typedef struct eddy Eddy;

Eddy *eddies;

struct cell{
	double x;
	double y;
	double vx;
	double vy;
	double Cn;
	double Cn0;
	double Cp;
	double Cz;
	double Fx;//chemotaxis drift
	double Fy;
};typedef struct cell Cell;

Cell **lattice,**lattice0;



//////////////////////////////////////////////////////
////////////////// HELPERS ///////////////////////////
//////////////////////////////////////////////////////


const gsl_interp2d_type *inter_type = gsl_interp2d_bilinear;         /* number of points to interpolate */
const double xa[] = { 0.0, 1.0 }; /* define unit square */
const double ya[] = { 0.0, 1.0 };
const size_t nx = sizeof(xa) / sizeof(double); /* x grid points */
const size_t ny = sizeof(ya) / sizeof(double); /* y grid points */
double *za = (double *)malloc(nx * ny * sizeof(double));
gsl_spline2d *spline = gsl_spline2d_alloc(inter_type, nx, ny);
gsl_interp_accel *xacc = gsl_interp_accel_alloc();
gsl_interp_accel *yacc = gsl_interp_accel_alloc();	



void FFT1D(fftw_complex *in, fftw_complex *out, int L){
	fftw_plan my_plan;
	my_plan = fftw_plan_dft_1d(L, in, out, FFTW_FORWARD,FFTW_ESTIMATE);
	fftw_execute(my_plan);
	fftw_destroy_plan(my_plan);
}

void iFFT1D(fftw_complex *in,fftw_complex *out, int L){
	fftw_plan my_plan;
	my_plan = fftw_plan_dft_1d(L, in, out, FFTW_BACKWARD,FFTW_ESTIMATE);
	fftw_execute(my_plan);
	fftw_destroy_plan(my_plan);
}

void getY_E(){		
	#pragma omp parallel for
	for(int i=0;i<n_eddies;i++){
		y_eddies[i] = (eddies[i].x);
		y_eddies[i+n_eddies] = (eddies[i].y);			
	}
}


void getE_Y(){
	int j=0;
	int nt=n_eddies+n_phyto+n_zoo;
	#pragma omp parallel private(j)
	for(int i=0;i<n_eddies;i++){		
		eddies[i].x = y_eddies[i];
		eddies[i].y = y_eddies[i+n_eddies];		
	}
}

/// GEOMETRY ////
double pos_x(int i){
	int id = i%(L*L);
	int xi = id%L;	
	return (xi-L/2)*dx;
}
double pos_y(int i){
	int id = i%(L*L);
	int yi = int(id/L);	
	return (yi-L/2)*dx;
}

double pfc(int i,int j){
	return sqrt( dx*dx*(i-L/2)*(i-L/2) + dx*dx*(j-L/2)*(j-L/2) );
}

double periodic(double x){
	if(x>L-1){
		return x-L;
	}
	if(x<0){
		return L+x;
	}
	return x;
}

double dist(double a,double b){//primary
	double d=(a-b);
	double dp=d;
	if(d>L/2){
		dp = -(L-d);
	}
	else if(d<-L/2){
		dp = (L+d);
	}
	return dp;	
}
double dist2(double a,double b){//complementary
	double dp = dist(a,b);
	return -(L-abs(dp))*GSL_SIGN(dp);
}
double dist3(double a,double b){//primary looped
	double dp = dist(a,b);
	return dp + GSL_SIGN(dp)*L;
}
double dist4(double a,double b){//complementary looped
	double dc = dist2(a,b);
	return dc + GSL_SIGN(dc)*L;
}


double chiDistance(double dx, double dy, double R){
	double dist = sqrt(dx*dx + dy*dy);
	return pow(fabs(dist - R)/R,2);
}


double getAngle(double x1, double y1, double x2, double y2){
	double m = sqrt(x1*x1 + y1*y1)*sqrt(x2*x2 + y2*y2);
	return acos((x1*x2 + y1*y2)/m);
}



/// FLOW ADVECTION EQUATION ///

void getFlowVelocity(double x,double y,double *vx_e,double *vy_e){
	double vx0=0;
	double vy0=0;
	double vx=0;
	double vy=0;
	double xa,ya;
	double omega;	
	double x0=0;
	double y0=0;
	double r2=0;

	#pragma omp parallel for reduction(+:vx,vy)
	for(int j=0;j<n_eddies;j++){		
		x0=dist(x,eddies[j].x);
		y0=dist(y,eddies[j].y);
		if(x0!=0 || y0!=0){
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);		
		}
		//complementary distance
		x0=dist2(x,eddies[j].x);
		y0=dist2(y,eddies[j].y);
		if(x0>0 || y0>0){
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);
		}	
		//// primary looped
		x0=dist3(x,eddies[j].x);
		y0=dist3(y,eddies[j].y);	
		if(x0>0 || y0>0){	
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);
		}	
		// complementary looped
		x0=dist4(x,eddies[j].x);
		y0=dist4(y,eddies[j].y);
		if(x0>0 || y0>0){
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);
		}	
	}
	//rotation by 90 counter-clockwise	
	*vx_e = vx;
	*vy_e = vy;		
	return;
}


void getFlowVelocity_ode(double x,double y,double *vx_e,double *vy_e,const double y_ode0[],int nt){
	double vx0=0;
	double vy0=0;
	double vx=0;
	double vy=0;
	double xa,ya;
	double omega;	
	double x0=0;
	double y0=0;
	double r2=0;

	#pragma omp parallel for reduction(+:vx,vy)
	for(int j=0;j<n_eddies;j++){
		x=periodic(x);
		y=periodic(y);
		eddies[j].x=periodic(y_ode0[j]);		
		eddies[j].y=periodic(y_ode0[j+nt]);		
		x0=dist(x,eddies[j].x);
		y0=dist(y,eddies[j].y);
		if(x0!=0 || y0!=0){
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);		
		}
		//complementary distance
		x0=dist2(x,eddies[j].x);
		y0=dist2(y,eddies[j].y);
		if(x0>0 || y0>0){
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);		

		}	
		//// primary looped
		x0=dist3(x,eddies[j].x);
		y0=dist3(y,eddies[j].y);	
		if(x0>0 || y0>0){	
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);		
		}	
		// complementary looped
		x0=dist4(x,eddies[j].x);
		y0=dist4(y,eddies[j].y);
		if(x0>0 || y0>0){
			r2=(eddies[j].R*eddies[j].R);
			vx+=a_eddy*eddies[j].w*y0*exp(-(x0*x0 + y0*y0)/r2);
			vy+=-a_eddy*eddies[j].w*x0*exp(-(x0*x0 + y0*y0)/r2);		

		}	
	}
	//rotation by 90 counter-clockwise	
	*vx_e = vx;
	*vy_e = vy;		
	return;
}


int func_advectY (double t, const double y[], double f[],void *params){
	(void)(t); /* avoid unused parameter warning */	
	int j=0;
	int nt = n_eddies+n_phyto+n_zoo;
	//double mu = *(double *)params;
	double x1,y1;
	double vx,vy;

	#pragma omp parallel for 
	for(int j=0;j<nt;j++){
		x1=y[j];
		y1=y[j+nt];	
		getFlowVelocity_ode(x1,y1,&vx,&vy,y,nt);		
		f[j] = vx;
		f[j+nt] = vy;							
	}
	
	// scanf("%d",&n_phyto);

	return GSL_SUCCESS;
}

double DIFFUSIONp(int i, int j){		
	return D*(-4*lattice[i][j].Cp+lattice[int(periodic(i+1))][j].Cp+lattice[int(periodic(i-1))][j].Cp+lattice[i][int(periodic(j+1))].Cp+lattice[i][int(periodic(j-1))].Cp);
}
double DIFFUSIONz(int i, int j){		
	return D*(-4*lattice[i][j].Cz+lattice[int(periodic(i+1))][j].Cz+lattice[int(periodic(i-1))][j].Cz+lattice[i][int(periodic(j+1))].Cz+lattice[i][int(periodic(j-1))].Cz);
}
double DIFFUSIONn(int i, int j){		
	return D*(-4*lattice[i][j].Cn+lattice[int(periodic(i+1))][j].Cn+lattice[int(periodic(i-1))][j].Cn+lattice[i][int(periodic(j+1))].Cn+lattice[i][int(periodic(j-1))].Cn);
}


double FitZAP(double v){
	double DELTA_FIT = (1+ALPHA*pow(v,-BETA))/(z_cost_0*z_death_rate) - z_benefits_0*1.0/(z_death_rate);	
	if(Z_toggle==0){
		return strat;
	}	
	if(DELTA_FIT>0){
		return 1; //active
	}
	else{
		return 0; //passive
	}
}

double ZCOST(double v){
	if(FitZAP(v)==0){
		return 1;
	}
	else{
		return z_cost_0;
	}
}

double ZBENF(double v){
	if(FitZAP(v)==1){
		return 1;
	}
	else{
		return z_benefits_0;
	}
}

double ZACT(double v){	
	if(FitZAP(v)==0){
		return 0;
	}
	else{
		return 1;
	}
}

double G(double u, double v){	
	return ZBENF(v)*M_PI*z_catching_rate*z_catching_range*z_catching_range*(v + ZACT(v)*ALPHA*pow(v,1.0-BETA));
}


double FP(double v,double u, double w){	
	if(u<0){
		u=0;
	}
	if(v<0){
		v=0;
	}
	if(w<0){
		w=0;
	}
	return p_reproduction_rate*v*(1-v/w) - G(u,v)*u;
}
double FZ(double v,double u,double w){
	if(u<0){
		u=0;
	}
	if(v<0){
		v=0;
	}
	if(w<0){
		w=0;
	}
	return -ZCOST(v)*z_death_rate*u + z_reproduction_rate*G(u,v)*u;
}
double FN(double v,double u,double w,double w0){	
	if(u<0){
		u=0;
	}
	if(v<0){
		v=0;
	}
	if(w<0){
		w=0;
	}
	return -gammar*(w-w0);
}

int func_DensY (double t, const double y[], double f[],void *params){
	(void)(t); /* avoid unused parameter warning */	


	f[0]=FP(y[0],y[1],y[2]);
	f[1]=FZ(y[0],y[1],y[2]);
	f[2]=FN(y[0],y[1],y[2],y[3]);
	f[3]=0.0;
	
	return GSL_SUCCESS;
}

////////////////////////////
/// GET AND WRITE ///
double getField(){
	double x,y,vx,vy;
	double xd,yd;
	double mv=0;
	#pragma omp parallel shared(vectorGrid) private(x,y,vx,vy)
	{			
		#pragma omp for collapse(2) reduction(+:mv)
		for(int i=0; i<L;i++){			
			for(int j=0;j<L;j++){				
				x=i*dx;
				y=j*dx;				
				getFlowVelocity(x,y,&vx,&vy);						
				vectorGrid[i][j][0]=vx;
				vectorGrid[i][j][1]=vy;					
				mv+=sqrt(vx*vx+vy*vy)/(L*L);				
			}
		}
	}		
	return mv;
}

double* getSpectrum(double** FF){
	fftw_complex *vF,*vF_ft;	
	vF = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*L);
	vF_ft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*L);	
	double* Sc = (double *)malloc(sizeof(double)*L);
	for(int i=0;i<L;i++){
		Sc[i]=0;
	}
	for(int i=0; i<L;i++){		
		double x,y,vx,vy;	
		#pragma omp parallel for	
		for(int j=0;j<L;j++){				
			x=i*dx;
			y=j*dx;									
			vF[j][0] = FF[i][j];	
			vF[j][1] = 0;							
		}				
		FFT1D(vF,vF_ft,L);		
		#pragma omp parallel for
		for(int j=0;j<L;j++){			
			Sc[j] += ((vF_ft[j][0]*vF_ft[j][0] + vF_ft[j][1]*vF_ft[j][1])*(dx))/(L);				
		}	
	}	
	free(vF);
	free(vF_ft);
	return Sc;
}

void writeAllSpectra(){
	double ** s;
	s = (double **)malloc(sizeof(double*)*(L)); // 2 species
	for(int i=0;i<L;i++){
		s[i]=(double *)malloc(sizeof(double)*(L)); 
	}
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			s[i][j]=lattice[i][j].Cp;
		}
	}
	Sv = getSpectrum(s);
	//
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			s[i][j]=lattice[i][j].Cz;
		}
	}
	Su = getSpectrum(s);
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			s[i][j]=lattice[i][j].Cn;
		}
	}
	Sw = getSpectrum(s);
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			s[i][j]=vectorGrid[i][j][0];
		}
	}
	Sf = getSpectrum(s);
	
	for(int i=0;i<L;i++){
		fprintf(fspectra,"%f\t%f\t%.17g\t%.17g\t%.17g\t%.17g\n",T,2.0*i/L,Sf[i],Sv[i],Su[i],Sw[i]);
	}	
	fprintf(fspectra,"\n");
}


void getFieldSpectrum(){
	fftw_complex *vF,*vF_ft;	
	vF = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*L);
	vF_ft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*L);	
	for(int i=0; i<L;i++){		
		double x,y,vx,vy;	
		#pragma omp parallel for	
		for(int j=0;j<L;j++){				
			x=i*dx;
			y=j*dx;									
			vF[j][0] = vectorGrid[i][j][0];	
			vF[j][1] = 0;				
		}				
		FFT1D(vF,vF_ft,L);		
		#pragma omp parallel for
		for(int j=0;j<L;j++){			
			vecField_ft_avg[j]+=((vF_ft[j][0]*vF_ft[j][0] + vF_ft[j][1]*vF_ft[j][1])*(dx))/(L);				
		}	
	}	
	free(vF);
	free(vF_ft);
}



void getSmoothness(double* smo){
	double m[3];
	double std[3];
	for(int i=0;i<3;i++){
		m[i]=0;
		std[i]=0;
	}
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			if(lattice[i][j].Cp>0)
				m[0] = m[0] + fabs(DIFFUSIONp(i,j)/D)/(L*L);
			if(lattice[i][j].Cz>0)
				m[1] = m[1] + fabs(DIFFUSIONz(i,j)/D)/(L*L);
			if(lattice[i][j].Cn>0)
				m[2] = m[2] + fabs(DIFFUSIONn(i,j)/D)/(L*L);
		}
	}
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			if(lattice[i][j].Cp>0)
				std[0] = std[0] + fabs(fabs(DIFFUSIONp(i,j)/D-m[0]))/(L*L);
			if(lattice[i][j].Cz>0)
				std[1] = std[1] + fabs(fabs(DIFFUSIONz(i,j)/D-m[1]))/(L*L);
			if(lattice[i][j].Cn>0)
				std[2] = std[2] + fabs(fabs(DIFFUSIONn(i,j)/D-m[2]))/(L*L);
		}
	}
	for(int i=0;i<3;i++){
		smo[i]=std[i]/m[i];
	}
}



void getSmoothnessMEAN(double* smo){
	double m[3];
	double std[3];
	for(int i=0;i<3;i++){
		m[i]=0;
		std[i]=0;
	}
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			if(lattice[i][j].Cp>0)
				m[0] = m[0] + fabs(DIFFUSIONp(i,j)/D)/(L*L);
			if(lattice[i][j].Cz>0)
				m[1] = m[1] + fabs(DIFFUSIONz(i,j)/D)/(L*L);
			if(lattice[i][j].Cn>0)
				m[2] = m[2] + fabs(DIFFUSIONn(i,j)/D)/(L*L);
		}
	}

	for(int i=0;i<3;i++){
		smo[i]=m[i];
	}
}


void writeField(){
	double x,y;
	double smo[3];
	getSmoothnessMEAN(smo);
	for(int i=0; i<L;i++){			
		for(int j=0;j<L;j++){			
			fprintf(fvector, "%.17g\t%f\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n", T,LCUTOFF,i,j,lattice[i][j].Cp,lattice[i][j].Cz,FitZAP(lattice[i][j].Cp),lattice[i][j].Cn,fabs(DIFFUSIONp(i,j)/D)/smo[0],fabs(DIFFUSIONz(i,j)/D)/smo[1],fabs(DIFFUSIONn(i,j)/D)/smo[2],vectorGrid[i][j][0],vectorGrid[i][j][1],sqrt(vectorGrid[i][j][0]*vectorGrid[i][j][0]+vectorGrid[i][j][1]*vectorGrid[i][j][1]));
		}
	}
	fprintf(fvector, "\n\n");
	fflush(fvector);	
}



//////////////////////////////////////////////////////
////////////////// FUNCTIONS ///////////////////////////
//////////////////////////////////////////////////////

void initialize(){
    /////////////////////
    ///////////////////
    strcpy(file4_name,"2DpdeADAP_v2_bloom_field");   
    strcpy(file2_name,"2DpdeADAP_v2_bloom_spectrum");       
    strcpy(file3_name,"2DpdeADAP_v2_bloom_evolution");
    strcpy(file1_name,"2DpdeADAP_v2_bloom_distribution");    
    //
    sprintf(file_info,"_%d_%f_%f_%f_%f_%f_%d.dat",L,zoo_V0,a_eddy,z_cost_0,z_benefits_0,z_catching_rate,seed);
    //
    strcat(file1_name,file_info);
    strcat(file2_name,file_info);
    strcat(file3_name,file_info);
    strcat(file4_name,file_info);    

    fdistribution = fopen(file1_name,"w+");
    fspectra = fopen(file2_name,"w+");
    fevolution = fopen(file3_name,"w+");
    fvector = fopen(file4_name,"w+");

	vecField_ft_avg = (double *)malloc(sizeof(double)*resolution);	
	vectorGrid = (double ***)malloc(sizeof(double**)*resolution);

	flowDeltaVelocities = (double *)malloc(sizeof(double)*n_eddies*(n_eddies+1)/2);

	for(int i=0;i<resolution;i++){
		vecField_ft_avg[i]=0;				
		vectorGrid[i] = (double **)malloc(sizeof(double*)*resolution);
	}
	for(int i=0;i<resolution;i++){
		for(int j=0;j<resolution;j++){
			vectorGrid[i][j] = (double *)malloc(sizeof(double)*2);
		}
	}

	eddies = (Eddy *)malloc(sizeof(Eddy)*n_eddies);
	y_eddies = (double *)malloc(sizeof(double)*(2*(n_eddies)));	

	lattice = (Cell **)malloc(sizeof(Cell*)*(L)); // 2 species
	for(int i=0;i<L;i++){
		lattice[i]=(Cell *)malloc(sizeof(Cell)*(L)); 
	}
	lattice0 = (Cell **)malloc(sizeof(Cell*)*(L)); // 2 species
	for(int i=0;i<L;i++){
		lattice0[i]=(Cell *)malloc(sizeof(Cell)*(L)); 
	}

	if(y_eddies == NULL || lattice == NULL || lattice0 == NULL){
		printf("allocation error\n");
	}

	return;
}

void initialCondition(){
	double theta=0;
	double de=0.05;
	double c0=0;
	dt=dt_0;	
	T=0;

	///////
	gsl_rng_set(r,seed);

	for(int i=0;i<n_eddies;i++){		
		eddies[i].x=gsl_rng_uniform(r)*L;//gsl_rng_uniform(r)*L;
		eddies[i].y=gsl_rng_uniform(r)*L;//gsl_rng_uniform(r)*L;
		//printf("%lf\t%lf\n", eddies[i].x,eddies[i].y);
		eddies[i].vx=0.0;
		eddies[i].vy=0.0;
		if(gsl_rng_uniform(r)<0.5){
			eddies[i].w=1.0;
		}
		else{
			eddies[i].w=-1.0;
		}
		if(R_min==R_max){
			eddies[i].R=R_min;
		}
		else{
			eddies[i].R=pow(((pow(R_max,-pexp+1) - pow(R_min,-pexp+1))*gsl_rng_uniform(r) + pow(R_min,-pexp+1)),1.0/(-pexp+1));
		}
		fflush(stdout);
	}
	getY_E();
	//////

	double ceff = M_PI*z_catching_rate*z_catching_range*z_catching_range;
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			lattice[i][j].x=i*dx;
			lattice[i][j].y=j*dx;
			lattice[i][j].vx=0;
			lattice[i][j].vy=0;
			lattice[i][j].Cn=10000*exp(-float( (j-L/2)*(j-L/2) )*(LCUTOFF/100000)*(LCUTOFF/100000)/(100));
			lattice[i][j].Cn0=lattice[i][j].Cn;
			lattice[i][j].Cp=1.0*exp(-float( (j-L/2)*(j-L/2) )*(LCUTOFF/100000)*(LCUTOFF/100000)/(100));
			lattice[i][j].Cz=1e-6;	
			total_P+=lattice[i][j].Cp*dx*dx;
			total_Z+=lattice[i][j].Cz*dx*dx;
			total_N+=lattice[i][j].Cn*dx*dx;
		}
	}

	sys_eddy = {func_advectY, NULL, size_t(2*(n_eddies)), NULL};
	d_eddy = gsl_odeiv2_driver_alloc_y_new(&sys_eddy, gsl_odeiv2_step_rk2, dt_0, ode_epsabs, 0.0); 
    
   	sys_dens = {func_DensY, NULL, size_t(4), NULL};
	d_dens = gsl_odeiv2_driver_alloc_y_new(&sys_dens, gsl_odeiv2_step_rkf45, dt_0, 0.001 , 0.0); 
    
	
	

	///////////////////
	//GET INITIAL FLOW FIELD AND FEATURES
	//////////////////////////////////////////
	avgV=getField();
	printf("%f\n", avgV);
	//getFieldSpectrum();		
	//writeField();	
	printf("allocated and initialized\n");
}

void Advect_Eddies(){
	int s;
	getY_E();
	////
	s = gsl_odeiv2_driver_apply (d_eddy, &T, T+dt, y_eddies);	
	if (s != GSL_SUCCESS){
		printf ("error: driver returned %d\n", s);
	}
	for(int i=0;i<2*n_eddies;i++){
		y_eddies[i]=periodic(y_eddies[i]);		
	}		
	getE_Y();
}

void chemoDirection(int i,int j,double *VX,double *VY){
	double vv;
	double vx = (lattice[(int)periodic(i+1)][j].Cp-lattice[(int)periodic(i-1)][j].Cp)/(lattice[i][j].Cp) ;
	double vy = (lattice[(int)periodic(i)][(int)periodic(j+1)].Cp-lattice[(int)periodic(i)][(int)periodic(j-1)].Cp)/(lattice[i][j].Cp);	
	//printf("%lf\t%lf\n", vx,vy);
	*VX=0;
	*VY=0;
	vx = vx + gsl_ran_gaussian(r,1)*sqrt(vx);
	vy = vy + gsl_ran_gaussian(r,1)*sqrt(vy);	
	vv = sqrt(vx*vx + vy*vy);	
	if(vx == 0 && vy==0){
		*VX =  0;
		*VY =  0;	
	}
	else{
		*VX =  vx/vv;
		*VY =  vy/vv;		
	}	
	*VX=0;	
	*VY=0;
}




void Advect_Species(){

	double fx,fy;
	double cp=0;
	double cz=0;
	double cn=0;
	double rv,tv;
	int bx[4],by[4];	
	double VX=0;
	double VY=0;		


	double t1=0;
	int s = 0;
	double ydense[4];	

	
	double total_P0=total_P;
	double total_N0=total_N;
	double total_Z0=total_Z;
	

	total_P=0;
	total_N=0;
	total_Z=0;

	avgV=getField();

	double fP0;
	double fN0;
	double fZ0;

	
	//#pragma omp parallel for reduction(+:total_P,total_Z,total_N) shared(vectorGrid)
	for(int i=0;i<L;i++){
		for(int j=0;j<L;j++){
			//cp flow
			fx = i*dx - (vectorGrid[i][j][0])*dt;
			fy = j*dx - (vectorGrid[i][j][1])*dt;
			bx[0] = periodic(floor(fx));
			by[0] = periodic(floor(fy));
			bx[1] = periodic(floor(fx)+1);
			by[1] = periodic(floor(fy));
			bx[2] = periodic(floor(fx));
			by[2] = periodic(floor(fy)+1);
			bx[3] = periodic(floor(fx)+1);
			by[3] = periodic(floor(fy)+1);
			fx = fx-floor(fx);
			fy = fy-floor(fy);



			gsl_spline2d_set(spline, za, 0, 0, lattice[bx[0]][by[0]].Cp );
			gsl_spline2d_set(spline, za, 1, 0, lattice[bx[1]][by[1]].Cp );
			gsl_spline2d_set(spline, za, 0, 1, lattice[bx[2]][by[2]].Cp );
			gsl_spline2d_set(spline, za, 1, 1, lattice[bx[3]][by[3]].Cp );

			/* initialize interpolation */
			gsl_spline2d_init(spline, xa, ya, za, nx, ny);
			cp=gsl_spline2d_eval(spline, fx, fy, xacc, yacc);	

			//cz flow
			chemoDirection(i,j,&VX,&VY);			
			fx = i*dx - (vectorGrid[i][j][0] + VX*DFT_fx)*dt;
			fy = j*dx - (vectorGrid[i][j][1] + VY*DFT_fx)*dt;
			bx[0] = periodic(floor(fx));
			by[0] = periodic(floor(fy));
			bx[1] = periodic(floor(fx)+1);
			by[1] = periodic(floor(fy));
			bx[2] = periodic(floor(fx));
			by[2] = periodic(floor(fy)+1);
			bx[3] = periodic(floor(fx)+1);
			by[3] = periodic(floor(fy)+1);
			fx = fx-floor(fx);
			fy = fy-floor(fy);		

			gsl_spline2d_set(spline, za, 0, 0, lattice[bx[0]][by[0]].Cz );
			gsl_spline2d_set(spline, za, 1, 0, lattice[bx[1]][by[1]].Cz );
			gsl_spline2d_set(spline, za, 0, 1, lattice[bx[2]][by[2]].Cz );
			gsl_spline2d_set(spline, za, 1, 1, lattice[bx[3]][by[3]].Cz );

			gsl_spline2d_init(spline, xa, ya, za, nx, ny);
			cz=gsl_spline2d_eval(spline, fx, fy, xacc, yacc);

			fx = i*dx - (vectorGrid[i][j][0])*dt;
			fy = j*dx - (vectorGrid[i][j][1])*dt;
			bx[0] = periodic(floor(fx));
			by[0] = periodic(floor(fy));
			bx[1] = periodic(floor(fx)+1);
			by[1] = periodic(floor(fy));
			bx[2] = periodic(floor(fx));
			by[2] = periodic(floor(fy)+1);
			bx[3] = periodic(floor(fx)+1);
			by[3] = periodic(floor(fy)+1);
			fx = fx-floor(fx);
			fy = fy-floor(fy);


			gsl_spline2d_set(spline, za, 0, 0, lattice[bx[0]][by[0]].Cn );
			gsl_spline2d_set(spline, za, 1, 0, lattice[bx[1]][by[1]].Cn );
			gsl_spline2d_set(spline, za, 0, 1, lattice[bx[2]][by[2]].Cn );
			gsl_spline2d_set(spline, za, 1, 1, lattice[bx[3]][by[3]].Cn );

			/* initialize interpolation */
			//printf("%lf\t%lf\n", za[0],za[1]);
			gsl_spline2d_init(spline, xa, ya, za, nx, ny);
			cn=gsl_spline2d_eval(spline, fx, fy, xacc, yacc);

			lattice[i][j].Cp=cp;
			lattice[i][j].Cz=cz;
			lattice[i][j].Cn=cn;
				

			if(lattice[i][j].Cp < 0)
				lattice[i][j].Cp = 0;
			if(lattice[i][j].Cz < 0)
				lattice[i][j].Cz = 0;
			if(lattice[i][j].Cn < 0)
				lattice[i][j].Cn = 0;

			total_P+=lattice[i][j].Cp;
			total_Z+=lattice[i][j].Cz;
			total_N+=lattice[i][j].Cn;

		
		}
	}	
	fP0 = total_P0/total_P;
	fZ0 = total_Z0/total_Z;
	fN0 = total_N0/total_N;
	total_P=0;
	total_N=0;
	total_Z=0;	
	//#pragma omp parallel for reduction(+:total_P,total_Z,total_N)	
	for(int i=0;i<L;i++){	
		double ydense[4];	
		double t1=0;
		int s = 0;
		for(int j=0;j<L;j++){
			lattice[i][j].Cp=lattice[i][j].Cp*fP0;
			lattice[i][j].Cz=lattice[i][j].Cz*fZ0;
			lattice[i][j].Cn=lattice[i][j].Cn*fN0;				

			ydense[0]=lattice[i][j].Cp;
			ydense[1]=lattice[i][j].Cz;
			ydense[2]=lattice[i][j].Cn;
			ydense[3]=lattice[i][j].Cn0;
			gsl_odeiv2_driver_reset(d_dens);
			t1=0;
			s = gsl_odeiv2_driver_apply (d_dens, &t1, t1+dt, ydense);
			if (s != GSL_SUCCESS){
			 	printf ("error: driver returned %d\n", s);
			}

			double u=lattice[i][j].Cz;
			double v=lattice[i][j].Cp;
			double w=lattice[i][j].Cn;

			lattice[i][j].Cp=ydense[0] + DIFFUSIONp(i,j)*dt + sqrt(abs(DIFFUSIONp(i,j)))*gsl_ran_gaussian(r,1)*sqrt(dt)/LCUTOFF;
			lattice[i][j].Cz=ydense[1] + DIFFUSIONz(i,j)*dt + sqrt(abs(DIFFUSIONz(i,j)))*gsl_ran_gaussian(r,1)*sqrt(dt)/LCUTOFF;
			lattice[i][j].Cn=ydense[2] + DIFFUSIONn(i,j)*dt + sqrt(abs(DIFFUSIONn(i,j)))*gsl_ran_gaussian(r,1)*sqrt(dt)/LCUTOFF;

			lattice[i][j].Cp+=sqrt(dt)*gsl_ran_gaussian(r,1)*sqrt(abs(p_reproduction_rate*v*(1-v/w)) + G(u,v)*u)/(LCUTOFF);
			lattice[i][j].Cz+=sqrt(dt)*gsl_ran_gaussian(r,1)*sqrt(z_death_rate*u + z_reproduction_rate*G(u,v)*u)/(LCUTOFF);
			
			if(lattice[i][j].Cp<0){
				lattice[i][j].Cp=0;
			}
			if(lattice[i][j].Cz<0){
				lattice[i][j].Cz=0;
			}
			if(lattice[i][j].Cn<0){
				lattice[i][j].Cn=0;
			}

			total_P+=lattice[i][j].Cp;
			total_Z+=lattice[i][j].Cz;
			total_N+=lattice[i][j].Cn;	
		}
	}	
}

void DELTA(){	

	Advect_Eddies();	
	Advect_Species();

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////
/////////////////////////////
/////   MAIN    /////////////
/////////////////////////////


int main(int argc, char const *argv[])
{
	int count=0;
	int tid,nthreads;
	double smo[3];
	double f1,f2;
	///// RANDOM
	seed = 300 + atoi(argv[2]); //567
	L = 100;//time(NULL)%1000;
	zoo_V0=atof(argv[1]); // !!!!! in cm/s
	LCUTOFF=100*1000;//atof(argv[2]); // !!!!! in cm
    gsl_rng_env_setup();
    r = gsl_rng_alloc (gsl_rng_default);
    gsl_rng_set(r,seed);
    printf("seed: %d\n", seed);
    ///////////////////////////////////////
    /////// PREPARE ////////////////
    /////////////////////////

    DFT_p0 = 0.1*(1.0-exp(-3.40*zoo_V0));
	DFT_fx = 0.04*(1.0-exp(-2.57*zoo_V0))/LCUTOFF;

	
	strat=0;//atof(argv[2]);	
	Z_toggle = 1;//atoi(argv[1]);	
    a_eddy=0.6;//atof(argv[1]);//*LCUTOFF;//*(3600*24)/timescale; // 0.5
	n_eddies=1000;
	R_max=(float)L/4;
	R_min=(float)L/40;//0.0001 cm, 0.001 m, 1 km
	pexp=3.000;//4.3333;
	resolution=L;


	//a1 = 5.555668962927917;
	//a2 = 0.12095984043496572;
	psiD=0.25;
	D=(psiD*psiD/2 + DFT_p0*DFT_p0/2)/(LCUTOFF*LCUTOFF);	
	//8.40854586808467
	//0.151870026449854

	//NEWFIT
	a1=10.6100263282675;
	b1=2.32550880148483;
	a2=6.9945420110446;
	b2=1.66328721442047;

	f1=a1*exp(-psiD*b1);
	f2=a2*pow(psiD,(b2));	

	ALPHA = f1*(1-exp(-zoo_V0/f2));
	BETA = 1.00;

    printf("alfa %lf\t vz %lf\n", ALPHA,zoo_V0);


	#pragma omp parallel
	{
		int tid = omp_get_thread_num();	
		if (tid == 0) 
		{			
		 	printf("Number of threads = %d\n", omp_get_num_threads());
		}				
	}
	printf("initiate..\n");
	initialize();
	printf("set IC\n");
	initialCondition();	
	/////////////////////
	printf("Simulation started\n");
	printf("%s\n", file_info);
	fflush(stdout);
	fflush(fspectra);	
	printf("%lf  km/day (%lf)\n",avgV,a_eddy); //avgV from km/day to cm/s
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////	
	//////////////////////   MAIN LOOP /////////////////////////////
	////////////////////////////////////////	
	// for(int i=0;i<resolution;i++){
	// 	fprintf(fspectra, "%.17g\t%.17g\n", 2.0*i/L,vecField_ft_avg[i]);
	// }		
	//writeAllSpectra();	
	while(T<total_time){							
		if(sub_time_2<=0){
			writeField();			
			writeAllSpectra();
			sub_time_2=saveData_time_2;
		}
		else{
			sub_time_2=sub_time_2-dt;
		}
		if(sub_time_1<=0){
			getSmoothness(smo);
			fprintf(fevolution, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", T,total_P,total_Z,total_N,smo[0],smo[1],smo[2],avgV);
			fflush(fevolution);
			sub_time_1=saveData_time_1;
		}
		else{
			sub_time_1=sub_time_1-dt;
		}		
		DELTA();
		T=T+dt;//	TIME IS SHIFTED BY gsl.ode and 'dt' might change due to BirthDeath()
	}		
	////////////////////////////////////////////////////////////

	////////////////////////////
	///// BORING STUFF /////////
	////////////////////////////	
	fclose(fdistribution);
	fclose(fspectra);
	fclose(fevolution);
	return 0;
}
