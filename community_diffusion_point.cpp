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
#include <omp.h>


double dt_0=0.1; //seconds
double dt=dt_0;
double dt_j=0;
double const dt_j0=0.1;
double ode_epsabs=0.1;
//// 
double deltaT=0;
double acc_dt=0;
double T=0;
double L=20.0; // cm
int trial=0;
int countT=0;

double TAU=25;
double ALPHA=0.25;


//float timescale=1.0/(3600*24); // from days to second
float timescale=1.0/(24*60*60); // per minute
double total_time=10000;//12*30*24*3600;//3600*24*12; // 30 days
float lengthscale=1;
double saveData_time_1 = 10;//60*60.0;//every hour
int saveData_time_2 = 500;//number of samples
//
double updateRates_time = 0;
double sub_time_1=saveData_time_1;
int sub_time_2=0;
int count_save=0;
//
float D=0.1;
//
float p_reproduction_rate=timescale*1.0;
float p_death_rate=0;
float p_carrying_capacity=1000;
float z_reproduction_rate=0.01;
float z_catching_rate=timescale*1.0;
float z_bodysize=0.10/lengthscale; // cm estimated spherical radius
float z_catching_range=2*z_bodysize/lengthscale; //catching range/radius
float z_perceptual_range=10*z_bodysize/lengthscale; //0.5 cm body size - zooplankton
float z_death_rate=timescale*0.1;
float p_perceptual_range=0.01; //0.0005 cm body size
float p_bodysize=0.01; //0.0005 cm body size
float z_b0=10000000; //par of the response function of the catching rate
//double zoo_jumprate=1.0;//10*timescale*(24*3600);

double rateTOT=0;
double birthTOT=0;
double deathTOT=0;
double pbirthTOT=0;
double cRateTOT=0;

double total_C=0;
double total_P=0;
double total_Z=0;

double nextT_bd=0;

double velocityX_eddy=0;
double velocityY_eddy=0;
double a_eddy=32;//*(3600*24)/timescale;
//
double avgV=0;
double maxV=0;
double avgDistPP=0;
double avgDistZZ=0;
double avgDistZP=0;
int n_phyto_mf;
int n_zoo_mf;
int n_phyto_dp=0;
int n_zoo_dp=0;
int n_phyto_dm=0;
int n_zoo_dm=0;
int n_phyto_1;
int n_zoo_1;
int n_phyto_0;
int n_zoo_0;
double eff_c=1.0;

int N=1;
int n_eddies=1;
int n_particle=1000;
int n_phyto = 0;
int n_zoo = 0;
double dens_phyto=0;
double dens_zoo=0;
int n_max_particles=100000;
int n_max_ng=100;

double CHI=0.1;
double zoo_V0 = 1.0;

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


int n_trial=1;
double R_max=L/4;
double R_min=L/4;//0.0001 cm, 0.001 m, 1 km
double pexp=4.3333;
float dk = 0.05;//grid size
int resolution=int(L/(dk));
int Lres = resolution*resolution;
gsl_histogram * hP;
gsl_histogram * hZ;
double *dataE=0;
double *dataC=0;

double meanCatching=0;
double meanClumping=0;
double meanCatchingTarget=0;
double meanEncounterZP=0;
double meanEncounterChiZP=0;
double meanEncounterTargetZP=0;
double meanExperienceZ=0;
double meanShareP=0;
double meanCount=0;



double meanDistancePP=0;


int key=0;

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
char file5_name[99];
char file_info[99];

bool demogHasOccured=false;
bool jumpHasOccured=false;
double *vecField_ft_avg;
fftw_complex *flowSpectrum;
double ***vectorGrid;
double *flowDeltaVelocities;

int **gridS;
int ns=2;
int ls=5;


const int n_dens=14;
double dens[n_dens] = {0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24,20.48,40.96,81.92};

double *y_ode = NULL;
gsl_odeiv2_system sys;
gsl_odeiv2_driver * d;


struct eddy{
	double x;
	double y;
	double w;
	double R;
	double vx;
	double vy;
};typedef struct eddy Eddy;

Eddy *eddies;

struct particle{
	double x;
	double y;
	double vx;
	double vy;
	double birth_rate;
	double death_rate;
	double fx;
	double fy;
	double gx;
	double gy;
	double R;
	double Rchi;
	int Target_id;
	int nP;
	int nZ;
	int nchi;
	int *list_ng;
	int *list_chi;
};typedef struct particle Particle;

Particle *P;
Particle *Z;


//////////////////////////////////////////////////////
////////////////// HELPERS ///////////////////////////
//////////////////////////////////////////////////////

int p(int i, int j){
	return j*(resolution) + i;
}

int pt(int i, int j){	
	return j*(resolution) + i;
}


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





/// GEOMETRY ////
double periodic(double x){
	if(x>L){
		return fmod(x,L);
	}
	if(x<0){
		return L-fmod(-x,L);
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


double chiDistance(double dx, double dy, double R){
	double dist = sqrt(dx*dx + dy*dy);
	return pow(fabs(dist - R)/R,2);
}

double getAngle(double x1, double y1, double x2, double y2){
	double m = sqrt(x1*x1 + y1*y1)*sqrt(x2*x2 + y2*y2);
	return acos((x1*x2 + y1*y2)/m);
}

void AllPositionRandom(){
	for(int i=0;i<n_zoo;i++){
		Z[i].x = gsl_rng_uniform(r)*L;
		Z[i].y = gsl_rng_uniform(r)*L;
	}
	for(int i=0;i<n_phyto;i++){
		P[i].x = gsl_rng_uniform(r)*L;
		P[i].y = gsl_rng_uniform(r)*L;
	}
}

/// DEMOGRAPHY ///
void newP(int i){
	if(n_phyto+n_zoo<n_max_particles){
		P[n_phyto]=P[i];	
		double eta = gsl_rng_uniform(r)*(2*M_PI);
		P[n_phyto].x += p_bodysize*cos(eta);
		P[n_phyto].y += p_bodysize*sin(eta);
		n_phyto++;
	}	
}
void newP_ran(int i){
	if(n_phyto+n_zoo<n_max_particles){
		P[n_phyto]=P[i];			
		P[n_phyto].x = gsl_rng_uniform(r)*L;
		P[n_phyto].y = gsl_rng_uniform(r)*L;
		n_phyto++;
	}	
}
void newZ(int i){
	if(n_phyto+n_zoo<n_max_particles){
		if(Z[n_zoo].list_ng == NULL)
			Z[n_zoo].list_ng = (int *)malloc(sizeof(int)*n_max_ng);
		Z[n_zoo]=Z[i];
		double eta = gsl_rng_uniform(r)*(2*M_PI);
		Z[n_zoo].x += z_bodysize*cos(eta);
		Z[n_zoo].y += z_bodysize*sin(eta);
		n_zoo++;		
	}	
}
void newZ_ran(int i){
	if(n_phyto+n_zoo<n_max_particles){
		if(Z[n_zoo].list_ng == NULL)
			Z[n_zoo].list_ng = (int *)malloc(sizeof(int)*n_max_ng);
		Z[n_zoo]=Z[i];
		double eta = gsl_rng_uniform(r)*(2*M_PI);
		Z[n_zoo].x += gsl_rng_uniform(r)*L;
		Z[n_zoo].y += gsl_rng_uniform(r)*L;
		n_zoo++;		
	}	
}
void removeP(int i){
	P[i]=P[n_phyto-1];
	n_phyto--;
}
void removeZ(int i){
	Z[i]=Z[n_zoo-1];
	n_zoo--;
	//printf("- %d\n", i);
}
void catchPrey(int i){
	int j = Z[i].list_ng[int(0.99*gsl_rng_uniform(r)*Z[i].nP)]; // select prey in R
	Z[i].x=P[j].x; //jump to prey position
	Z[i].y=P[j].y;
	if(gsl_rng_uniform(r)<z_reproduction_rate){
		newZ(i);
	}
	removeP(j);	
	z_ingestionRate+=1.0/n_zoo;			
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
		for(int i=0; i<resolution;i++){			
			for(int j=0;j<resolution;j++){				
				vx=gsl_ran_gaussian(r,D);		
				vy=gsl_ran_gaussian(r,D);					
				mv+=sqrt(vx*vx+vy*vy)/(resolution*resolution);				
			}
		}
	}
	return mv;
}
void getFieldSpectrum(){
	fftw_complex *vF,*vF_ft;	
	vF = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*resolution);
	vF_ft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*resolution);	
	for(int i=0; i<resolution;i++){		
		double x,y,vx,vy;	
		#pragma omp parallel for	
		for(int j=0;j<resolution;j++){				
			x=i*dk;
			y=j*dk;	
			vx=gsl_ran_gaussian(r,D);		
			vy=gsl_ran_gaussian(r,D);								
			vF[j][0] = vx;	
			vF[j][1] = 0;				
		}				
		FFT1D(vF,vF_ft,resolution);		
		#pragma omp parallel for
		for(int j=0;j<resolution;j++){			
			vecField_ft_avg[j]+=((vF_ft[j][0]*vF_ft[j][0] + vF_ft[j][1]*vF_ft[j][1])*(dk))/(L*n_trial);				
		}	
	}	
	free(vF);
	free(vF_ft);
}
void writeField(){
	double x,y;
	for(int i=0; i<resolution;i++){			
		for(int j=0;j<resolution;j++){			
			x=i*dk;
			y=j*dk;				
			fprintf(fvector, "%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n", countT, T,x,y,vectorGrid[i][j][0],vectorGrid[i][j][1],sqrt(vectorGrid[i][j][0]*vectorGrid[i][j][0]+vectorGrid[i][j][1]*vectorGrid[i][j][1]));
		}
	}
	fprintf(fvector, "\n");
	fflush(fvector);	
}
void writePositions(){
	for(int i=0;i<n_phyto;i++){
		fprintf(fdistribution, "%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\n", countT,T, P[i].x,P[i].y,P[i].vx,P[i].vy,0);
	}
	for(int i=0;i<n_zoo;i++){
		fprintf(fdistribution, "%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n", countT,T, Z[i].x,Z[i].y,Z[i].vx,Z[i].vy,1,Z[i].gx,Z[i].gy,zoo_V0*Z[i].fx,zoo_V0*Z[i].fy,getAngle(Z[i].vx,Z[i].vy,zoo_V0*Z[i].fx,zoo_V0*Z[i].fy));
	}
	fprintf(fdistribution, "\n");
	fflush(fdistribution);
}



//////////////////////////////////////////////////////
////////////////// FUNCTIONS ///////////////////////////
//////////////////////////////////////////////////////

void initialize(){
    /////////////////////
    ////////////////////
    gsl_histogram_set_ranges_uniform(hP,0,L);
    gsl_histogram_set_ranges_uniform(hZ,0,L);

    // strcpy(file4_name,"2DcommunityDIFF_point_field");   
    // strcpy(file2_name,"2DcommunityDIFF_point_spectrum");       
    strcpy(file3_name,"2DcommunityPHASE_DBpoint");
    // strcpy(file1_name,"2DcommunityDIFF_point_distribution");
    // strcpy(file5_name,"2DcommunityDIFF_point_meanE");
    //
    sprintf(file_info,"_%f_%f_%f_%f_%1.0f.dat",D,zoo_V0,dens_phyto,dens_zoo,L);
    //
    strcat(file1_name,file_info);
    strcat(file2_name,file_info);
    strcat(file3_name,file_info);
    strcat(file4_name,file_info);
    strcat(file5_name,file_info);

    // fdistribution = fopen(file1_name,"w+");
    // fspectra = fopen(file2_name,"w+");
    fevolution = fopen(file3_name,"w+");
    // fvector = fopen(file4_name,"w+");
    // fhistogram = fopen(file5_name,"w+");


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
	P = (Particle *)malloc(sizeof(Particle)*n_max_particles);
	Z = (Particle *)malloc(sizeof(Particle)*n_max_particles);

	gridS = (int **)malloc(sizeof(int*)*ns);
	for(int i=0;i<ns;i++){
		gridS[i] = (int *)malloc(sizeof(int)*ns);
	}

	y_ode = (double *)malloc(sizeof(double)*(2*(n_eddies+n_max_particles)));
	if(y_ode == NULL){
		printf("allocation error\n");
	}

	dataE = (double *)malloc(sizeof(double)*saveData_time_2);
	dataC = (double *)malloc(sizeof(double)*saveData_time_2);

	return;
}

void initialCondition(){
	double theta=0;
	double de=0.05;
	double c0=0;
	dt=dt_0;
	countT=0;
	T=0;
	//////
	meanClumping=0;
	meanCatching=0;
	meanEncounterZP=0;
	meanEncounterChiZP=0;
	meanEncounterTargetZP=0;
	meanExperienceZ=0;
	meanShareP=0;
	meanCount=0;
	///////
	gsl_rng_set(r,seed);
	



	for(int i=0;i<n_max_particles;i++){
		free(P[i].list_ng);
		free(Z[i].list_ng);
	}

	for(int i=0;i<n_phyto;i++){
		theta=gsl_rng_uniform(r)*M_PI*2;
		de=gsl_rng_uniform(r);
		P[i].x=gsl_rng_uniform(r)*L;//L/2-0.2 + de*0.05*cos(theta);
		P[i].y=gsl_rng_uniform(r)*L;//0.5 + de*0.05*sin(theta);
		P[i].vx=0;
		P[i].vy=0;		
		P[i].birth_rate=p_reproduction_rate;
		P[i].death_rate=p_death_rate;
		P[i].R=p_perceptual_range;
		//P[i].list_ng = (int *)malloc(sizeof(int)*n_max_ng);
	}
	for(int i=0;i<n_zoo;i++){
		//theta=gsl_rng_uniform(r)*M_PI*2;
		//de=gsl_rng_uniform(r);
		Z[i].x=gsl_rng_uniform(r)*L;//L/2+0.2 + de*0.05*cos(theta);
		Z[i].y=gsl_rng_uniform(r)*L;//0.5 + de*0.05*sin(theta);
		Z[i].vx=0;
		Z[i].vy=0;		
		Z[i].birth_rate=0;//updated
		Z[i].death_rate=z_death_rate;
		Z[i].fx=0;
		Z[i].fy=0;
		Z[i].nP=0;
		Z[i].R=z_catching_range; // 1 cm
		Z[i].Rchi=z_perceptual_range;
		if(Z[i].list_ng==NULL)
			Z[i].list_ng = (int *)malloc(sizeof(int)*n_max_ng);
		if(Z[i].list_chi==NULL)
			Z[i].list_chi = (int *)malloc(sizeof(int)*n_max_ng);
		Z[i].nchi=0;
	}	

	deltaT = gsl_ran_exponential(r,1); // initiate Gillespie

	//////

		// ///////////////////
	// //GET INITIAL FLOW FIELD AND FEATURES
	// //////////////////////////////////////////
	// avgV=getField();
	// getFieldSpectrum();		
	// writeField();	
}

void getPerceptionZ(int i){
	double x=0;
	double y=0;
	double dx=0;
	double dy=0;
	double m=0;
	double m_count=0;
	int n=0;
	int nr=0;

	//#pragma omp parallel for reduction(+:n,nr,m) private(dx,dy)
	for(int j=0;j<n_phyto;j++){
		dx=dist(P[j].x,Z[i].x);
		dy=dist(P[j].y,Z[i].y);
		if(sqrt( dx*dx + dy*dy ) < Z[i].Rchi && n<n_max_ng){
			Z[i].list_chi[n]=j;
			n = n+1;	
			m = m + chiDistance(dx,dy,Z[i].Rchi);//weighted by distance
		}
		if(sqrt( dx*dx + dy*dy ) < Z[i].R && nr<n_max_ng){
			Z[i].list_ng[nr]=j;
			nr=nr+1;
		}
	}
	Z[i].nP=nr;
	Z[i].nchi=n;
	double ran = gsl_rng_uniform(r);		
	int j=0;
	if(Z[i].nchi>0){
		for(int k=0;k<Z[i].nchi;k++){
			j = Z[i].list_chi[k];
			dx=dist(P[j].x,Z[i].x);
			dy=dist(P[j].y,Z[i].y);
			m_count+=chiDistance(dx,dy,Z[i].Rchi)/m;
			if(m_count>ran){
		 		Z[i].gx=dx;
		 		Z[i].gy=dy;
		 		Z[i].Target_id=j;
		 		if(zoo_V0*dt_j0<sqrt(dx*dx+dy*dy)){
		 			Z[i].fx=dx/(sqrt(dx*dx+dy*dy));
			 		Z[i].fy=dy/(sqrt(dx*dx+dy*dy));	
			 	}
			 	else{
			 		Z[i].fx=dx/(zoo_V0*dt_j0);
			 		Z[i].fy=dy/(zoo_V0*dt_j0);
			 	}
			 	break;		
			}
		}
	}
	else{
		Z[i].fx=0;
		Z[i].fy=0;
	}

}


void getPerception(){	
	#pragma omp parallel for	
    for(int i=0;i<n_zoo;i++){
    	getPerceptionZ(i);        
    }
    return;
}

void ZooplanktonJumpAll(){		
	#pragma omp parallel for 
	for(int i=0;i<n_zoo;i++){				
		Z[i].x=periodic(Z[i].x + (zoo_V0*Z[i].fx)*dt_j0);
		Z[i].y=periodic(Z[i].y + (zoo_V0*Z[i].fy)*dt_j0);		
	}
	return;
}

void getRates(){
    birthTOT=0;
    deathTOT=0;
    rateTOT=0;
    double C=0;       

    getPerception();

    #pragma omp parallel for reduction(+:birthTOT)
    for(int i=0;i<n_zoo;i++){      
    	//RESPONSE FUNCTION
		Z[i].birth_rate=z_catching_rate*Z[i].nP;///(z_b0 + Z[i].nP);//(nPZ)/(z_b0+(nPZ));
		//Z[i].birth_rate=z_catching_rate*z_b0*Z[i].nP/(z_b0 + Z[i].nP);//(nPZ)/(z_b0+(nPZ));
        birthTOT += Z[i].birth_rate;
    }
    deathTOT += n_zoo*z_death_rate;
    C = (1.0-float(n_phyto)/(p_carrying_capacity*L*L));
    if(C<0){
    	pbirthTOT=0.0;
    	birthTOT += pbirthTOT;   
    	deathTOT += n_phyto*(p_death_rate) - C*n_phyto*p_reproduction_rate;    	
    }
    else{
    	pbirthTOT=C*n_phyto*(p_reproduction_rate);
    	birthTOT += pbirthTOT;   
    	deathTOT += n_phyto*(p_death_rate);
    }        

    rateTOT=birthTOT+deathTOT;    
}



////////////////////////////////////
////////////////////////////////////
////  MAIN FUNCTIONS ///////////////
////////////////////////////////////

double getTimeStep(){
	double new_dt = dt_0 - acc_dt;	
    ////////////////    
    /////
    getRates();
	if(rateTOT*new_dt <= deltaT){
		demogHasOccured=false;
		deltaT = deltaT - rateTOT*new_dt;
		/// complete dt_0 step				
		acc_dt = 0 ;
	}
	else if(rateTOT*new_dt > deltaT){ // a transition happens
		demogHasOccured=true;		
		new_dt = deltaT/(rateTOT);	
		deltaT = gsl_ran_exponential(r,1);
		acc_dt += new_dt;
	}
	////	
	////
	return new_dt;
}


void birthDeath(){
	double total_death=0;
	double total_birth=0;	
	double x=0;
    double ran=0;
    int id=0;
    double pbirth;
    double pdeath;
	double p=0;
	double nPZ=0;
	double xi=0;
	int n_zoo_0=n_zoo;
	int n_phyto_0=n_phyto;
	//printf("|| %lf\n", deltaT);

	if(demogHasOccured){			
		pbirth = birthTOT/rateTOT;
    	pdeath = deathTOT/rateTOT;		
		if(gsl_rng_uniform(r)<pbirth){			
			if(gsl_rng_uniform(r)<(pbirthTOT)/birthTOT){
				id=gsl_rng_uniform(r)*n_phyto;
				newP(id);
				//printf("birth P\n");
			}
			else{
				x=0;				
				ran=(birthTOT- pbirthTOT)*gsl_rng_uniform(r);
				for(int i=0;i<n_zoo;i++){
					x+=Z[i].birth_rate;
					if(ran<x){
						catchPrey(i);
						//printf("catch Z\n");
						break;
					}
				}
			}
		}
		else{
			id=gsl_rng_uniform(r)*n_zoo;
			removeZ(id);
			//printf("death Z\n");
		}	
	}
	return;
}


void AdvectSystem(){
	int s;
	int j=0;
	///////////////
	#pragma omp parallel for
	for(int i=0;i<n_phyto;i++){
		P[i].vx = gsl_ran_gaussian(r,D);
		P[i].vy = gsl_ran_gaussian(r,D);
		P[i].x = periodic(P[i].x+P[i].vx*sqrt(dt));
		P[i].y = periodic(P[i].y+P[i].vy*sqrt(dt));
	}
	#pragma omp parallel for
	for(int i=0;i<n_zoo;i++){
		Z[i].vx = gsl_ran_gaussian(r,D);
		Z[i].vy = gsl_ran_gaussian(r,D);
		Z[i].x = periodic(Z[i].x+Z[i].vx*sqrt(dt));
		Z[i].y = periodic(Z[i].y+Z[i].vy*sqrt(dt));
	}
}



////////////////////////////
// DATA ////////////////////
////////////////////////////


void getInfo(){
	double cz=0;
	double cp=0;
	double sp=0;
	double dx,dy;	
	int count=0;
	double nchi=0;
	double ncatching=0;
	double ntarget=0;
	double mv=0;
	int count_n=0;
	ClumpingZ=0;
	CatchingZ=0;
	shareP=0;
	ChiZ=0;

	#pragma omp parallel for reduction(+:nchi,ncatching,ntarget,mv) private(dx,dy,count_n)
	for(int i=0;i<n_zoo;i++){
		count_n=0;
		mv=mv+sqrt(Z[i].vx*Z[i].vx+Z[i].vy*Z[i].vy)/n_zoo;
		for(int j=0;j<n_phyto;j++){
			dx=dist(P[j].x,Z[i].x);
			dy=dist(P[j].y,Z[i].y);
			if(sqrt( dx*dx + dy*dy ) < Z[i].R){	
				count_n++;			
				ncatching=ncatching+double(1.00)/n_zoo;
				if(j==Z[i].Target_id){
					ntarget=ntarget+double(1.0)/n_zoo;
				}
				else{
					nchi=nchi+double(1.00)/n_zoo;
				}
			}
		}
		Z[i].nP=count_n;
	}	
	EncounterZP=ncatching;
	EncounterChiZP=nchi;
	EncounterTargetZP=ntarget;

	#pragma omp parallel for reduction(+:cz) private(dx,dy,count_n)
	for(int i=0;i<n_zoo;i++){
		count_n=0;
		for(int j=0;j<n_zoo;j++){
			dx = dist(Z[j].x,Z[i].x);
			dy = dist(Z[j].y,Z[i].y);
			if(sqrt(dx*dx+dy*dy)<z_catching_range && i!=j){
				count_n++;
				cz=cz+1.0/n_zoo;
			}
		}
		Z[i].nZ=count_n;
	}
	ClumpingZ=cz;	
	
}

void saveData(){

	n_phyto_0=n_phyto;
	n_zoo_0=n_zoo;
	
	if(sub_time_1<0 && T>200){
		sub_time_1=saveData_time_1; //1h
		getInfo();
		dataE[sub_time_2]=EncounterZP;
		dataC[sub_time_2]=ClumpingZ;
		sub_time_2+=1;
		//printf("%lf\n",EncounterZP);
		//fflush(stdout);
	}
	sub_time_1=sub_time_1-dt;
	if(sub_time_2>saveData_time_2){		
		//printf("%d\n",count_save);
		sub_time_2=0;
		fprintf(fevolution, "%f\t%f\t%f\t%f\t%f\t%.17g\t%.17g\t%.17g\t%.17g\n", D,zoo_V0,dens_phyto,dens_zoo,L,gsl_stats_mean(dataE,1,saveData_time_2),gsl_stats_variance(dataE,1,saveData_time_2),gsl_stats_mean(dataC,1,saveData_time_2),gsl_stats_variance(dataC,1,saveData_time_2));
		fflush(fhistogram);
		count_save=1;
	}
	//printf("%g\t%d\t%d\n", T,n_phyto,n_zoo);// time | n_phyto | n_zoo | catchingrate | clumping | sharing			

}
//////////////////////////
//////////////////////////
///// MASTER FUNCTION ////
//////////////////////////

void restoreParticles(){
	int nnp,nnz;
	nnp = n_phyto - n_phyto_0;
	nnz = n_zoo - n_zoo_0;	
	
	if(nnp>0){
		removeP(gsl_rng_uniform(r)*(n_phyto-1));
		n_phyto_dp++;		
	}	
	else if(nnp<0){
		newP_ran(gsl_rng_uniform(r)*(n_phyto-1));
		n_phyto_dm--;
	}
	if(nnz>0){
		removeZ(gsl_rng_uniform(r)*(n_zoo-1));
		n_zoo_dp++;
	}
	else if(nnz<0){
		newZ_ran(gsl_rng_uniform(r)*(n_zoo-1));
		n_zoo_dm--;
	}
}


void DELTA(){	
	saveData();
	//
	getPerception();
	dt=getTimeStep();//dt_0 otherwise what remains to a jump or demographic event		
	dt_j = dt_j + dt;
	if(dt_j >= dt_j0){
		ZooplanktonJumpAll();	
		dt_j=0;
	}							
	AdvectSystem();
	birthDeath();
	T=T+dt;	

	restoreParticles();

	//			
	//////////////////////////////////////////
	//EXTINCTION BREAK
	if(n_phyto==0 || n_zoo==0){
		T=total_time;
	}

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
	double x,y;
	double lower,upper;
	int tid,nthreads;	
	///// RANDOM
	seed = time(NULL)%100;
    gsl_rng_env_setup();
    r = gsl_rng_alloc (gsl_rng_default);
    gsl_rng_set(r,seed);
    //printf("seed: %d\n", seed);
    ////////////////////////////////////
    ///// ARGUMENTS    
    
    D=atof(argv[1]); //32
    zoo_V0=atof(argv[2]); //32    
    dens_phyto=atof(argv[3]);
    dens_zoo=atof(argv[4]);
    L=atof(argv[5]);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n_phyto=L*L*dens_phyto; //32
    n_zoo=L*L*dens_zoo; //32     
    printf("[%1.2f,%1.2f,%d,%d,%1.2f]\n",D,zoo_V0,n_phyto,n_zoo,L);

    ////////////////////////////////////////////////
	dk = 0.05;//grid size
	resolution=int(L/(dk));
	Lres = resolution*resolution;
	hP = gsl_histogram_alloc (resolution);
	hZ = gsl_histogram_alloc (resolution);

	saveData_time_2=20000/n_zoo;

    ////////////////////////////////////
    //////////////////////////////////
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();	
		if (tid == 0) 
		{			
		 	printf("Number of threads = %d\n", omp_get_num_threads());
		}				
	}	
	initialize();	
	initialCondition();		
	getRates();getPerception();	
	/////////////////////
	//printf("Simulation started\n");
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////	
	//////////////////////   MAIN LOOP /////////////////////////////
	////////////////////////////////////////		
	while(count_save<1){		
		DELTA();
		//T=T+dt;//	TIME IS SHIFTED BY gsl.ode and 'dt' might change due to BirthDeath()
	}	
	////////////////////////////////////////////////////////////

	////////////////////////////
	///// BORING STUFF /////////
	////////////////////////////
	free(Z);
	free(P);
	fclose(fevolution);
	return 0;
}
