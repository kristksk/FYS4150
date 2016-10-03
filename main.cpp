#include <iostream>
#include <armadillo>
#include <string>

using namespace std;
using namespace arma;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions for implementing the alogrithm
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void FindMaxElement(int ,int &, int &, mat &,double &);
void RotateTheMatrix(int , int &, int &, double &, double &, mat &, mat &);
void FindCosAndSin(int &, int &, double &, double &, mat &);
void Jacobi(int, mat & , uvec &diagonal_A, mat &U, int &);


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions to calculate the analytic result:
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void AnalyticSolutionTaut(int , double , vec &);


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions used to save results:
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void SaveToFile(string , string , string , vec , vec );
void SaveResultsToFile(string, int, int, int, double, mat, uvec, mat, uvec, vec, vec, double);


//~~~~~~~~~~~~~~~~~~~~~
// Unit test functions
//~~~~~~~~~~~~~~~~~~~~~~

void TestMaxElement();
void TestForEigenvalues();



int main()
{
    cout.precision(6); // To ensure precision in output

    //~~~~~~~~~~
    // Varibles:
    //~~~~~~~~~~

    int n = 200;
    double rho_max = 12;


    double h = rho_max / (n+2);
    double e = -1 / (h*h);
    double d = 2 / (h*h);


    //~~~~~~~~~~~~~~~~~~~
    // Constructing rho:
    //~~~~~~~~~~~~~~~~~~~

    vec rho(n+2);

    for(int i=0; i<n+2; i++){
        rho(i) += h*i;
    }
    rho(0) = 0;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // First: Single Electron Potential
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    vec V1 = rho % rho;
    mat A1 = zeros<mat>(n,n);

    for(int i=0; i<n; i++){
        A1(i,i) = d + V1(i+1);
    }

    A1.diag(1) += e;
    A1.diag(-1) += e;


    // Saving the matrix for further use with Armadillo
    mat A1_armadillo = A1;


    uvec diagonal_A1;
    mat U1 = eye<mat>(n,n);
    int iterations1;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Calling our Jacobi function,
    // thus running our algorithm for the single electron case.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Jacobi(n, A1, diagonal_A1, U1, iterations1);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Checking our algorithm by running the
    // Armadillo functions til solve the same problem
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mat eigvec1;
    vec eigval1;

    eig_sym(eigval1, eigvec1, A1_armadillo);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Secondly: Two Electron Potential
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    double omega_r = 0.25;
    vec V2 = omega_r*omega_r*(rho % rho) + 1 / rho;
    mat A2(n,n);
    for(int i=0; i<n; i++){
        A2(i,i) = d + V2(i+1);
    }

    A2.diag(1) += e;
    A2.diag(-1) += e;

    mat A2_armadillo = A2;

    uvec diagonal_A2;
    mat U2 = eye<mat>(n,n);
    int iterations2;
    Jacobi(n, A2, diagonal_A2, U2, iterations2);



    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Solving the two electron potential problem with
    // the Armadillo function:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mat eigvec2;
    vec eigval2;

    eig_sym(eigval2, eigvec2, A2_armadillo);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Calling the function of the analytic solution:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    vec Psi_Analytic;
    AnalyticSolutionTaut(n, h, Psi_Analytic);



    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Saving all our results in MATLAB- and txt-files
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Results from the Jacobi algorithm:

    SaveToFile("Jacobi_SingleElectron.m", "U", "rho", U1.col(diagonal_A1(0)), rho);
    SaveToFile("Jacobi_TwoElectrons.m", "U", "rho", U2.col(diagonal_A2(0)), rho);



    // Results from analytic solution

    SaveToFile("Analytic.m", "Psi", "rho", Psi_Analytic, rho);

    // Results from Armadillo

    SaveToFile("SingleElectron_Armadillo.m", "Psi", "rho", eigvec1.col(0), rho);
    SaveToFile("TwoElectrons_Armadillo.m", "Psi", "rho", eigvec2.col(0), rho);


    // Saving all results in a txt-file

    SaveResultsToFile("AllResults.txt", n, iterations1, iterations2, rho_max, A2, diagonal_A2, A1, diagonal_A1, eigval1, eigval2, omega_r);

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Functions implementing the algorithm  //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


void Jacobi(int n, mat &A, uvec &diagonal_A, mat &U, int &iterations){


    // Calling function to find max element in A:

    int k,l;
    double A_max = 0;
    FindMaxElement(n, k, l, A, A_max);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // Limits on the while-loop:      //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    iterations=0;
    int max_iterations = pow(10,6);
    double epsilon = pow(10,-9);



    while( fabs(A_max) > epsilon && (double) iterations < max_iterations){
        double c=0, s=0;
        FindCosAndSin(k, l, c, s, A);
        RotateTheMatrix(n, k, l, c, s, A, U);
        FindMaxElement(n, k, l, A, A_max);
        iterations++;

    }


    diagonal_A = sort_index(A.diag());
}

void FindMaxElement(int n,int &k, int &l, mat &A,double &A_max){

    A_max = 0.0;
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            if( fabs(A(i,j)) > A_max){
                k = i;
                l = j;
                A_max = fabs(A(i,j));
            }
        }
    }
}


void FindCosAndSin(int &k, int &l, double &c, double &s, mat &A){
    double tau, t, square_root;
    tau = -( A(l,l) - A(k,k) ) / ( 2 * A(k,l) );
    square_root = sqrt( 1 + tau*tau );
    if(tau < 0){
        t = tau + square_root;
    }
    else{
        t = tau - square_root;
    }

    c = 1 / sqrt( 1 + t*t);
    s = c*t;
}


void RotateTheMatrix(int n, int &k, int &l, double &c, double &s, mat &A, mat &U){

    for(int i=0; i<n; i++){
        if(i != k && i != l){
            A(i,i) = A(i,i);
            double a_il = A(i,l);
            double a_ik = A(i,k);
            A(i,k) = a_ik*c - a_il*s;
            A(i,l) = a_il*c + a_ik*s;

            // Because the matrix A is symmetric we can do the following:
            A(k,i) = A(i,k);
            A(l,i) = A(i,l);
        }

        // Eigenvectors are found:
        double u_ik = U(i,k);
        double u_il = U(i,l);
        U(i,k) = c*u_ik - s*u_il;
        U(i,l) = c*u_il + s*u_ik;
    }


    double a_kk = A(k,k);
    double a_ll = A(l,l);
    A(k,k) = a_kk*c*c - 2*A(k,l)*s*c + a_ll*s*s;
    A(l,l) = a_ll*c*c + 2*A(k,l)*s*c + a_kk*s*s;
    A(k,l) = 0;
    A(l,k) = 0;
}



void AnalyticSolutionTaut(int n, double h, vec &Psi_Analytic){

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // Analytic solution to compare results with  //
    // for the two electron potential.            //
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    vec u(n+2);
    int l=0;
    vec phi(n+2);


    for(int i=0; i<n+2; i++){
        u(i) += i*h;
    }

    u(0) = 0;
    Psi_Analytic = pow(u, 1) % exp(- pow(u, 2) / (8*(l+1)) ) % ( 1 + u / (2*(l+1) ) );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//Functions for saving our results to files (MATLAB and txt) //
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

void SaveToFile(string Filename, string vec_name1, string vec_name2, vec vec1, vec vec2){

    ofstream myfile;
    myfile.open (Filename);
    myfile << vec_name1 << "= [";
    for (int i=0; i<int(vec1.n_rows); i++){
        myfile << vec1(i) << ", ";
    }
    myfile << "];" << endl;

    myfile << vec_name2 << "= [";
    for (int i=0; i<int(vec2.n_rows); i++){
        myfile << vec2(i) << ", ";
    }
    myfile << "];" << endl;
    myfile << "plot("<< vec_name2 <<","<< vec_name1 << ".*"<< vec_name1 << ")" << endl;
    myfile << "n=" << vec2.n_rows << endl;
    myfile.close();
}


void SaveResultsToFile(string Filename, int n, int iterations1, int iterations2, double rho_max, mat A2, uvec diagonal_A2, mat A1, uvec diagonal_A1, vec eigval1, vec eigval2, double omega_r){
    ofstream myfile;
    myfile.open (Filename);
    myfile << "Single Electron Potential" << endl;
    myfile << "n: " << n << endl;
    myfile << "Number of iterations: " << iterations1 << endl;
    myfile << "rho_max: " << rho_max << endl;
    myfile << "lambda0: " << A1.diag()(diagonal_A1(0)) << "  lambda1: " << A1.diag()(diagonal_A1(1)) << "  lambda2: " << A1.diag()(diagonal_A1(2)) << endl;
    myfile << "Results from Armadillo:" << endl;
    myfile << "lambda0: " << eigval1(0) << "  lambda1: " << eigval1(1) << "  lambda2: " << eigval1(2) << endl;
    myfile << " " << endl;
    myfile << "Two Electron Potential:" << endl;
    myfile << "n: " << n << endl;
    myfile << "Number of iterations: " << iterations2 << endl;
    myfile << "rho_max: " << rho_max << endl;
    myfile << "omega_r: " << omega_r << endl;
    myfile << "lambda0: " << A2.diag()(diagonal_A2(0)) << "  lambda1: " << A2.diag()(diagonal_A2(1)) << "  lambda2: " << A2.diag()(diagonal_A2(2)) << endl;
    myfile << "Results from Armadillo:" << endl;
    myfile << "lambda0: " << eigval2(0) << "  lambda1: " << eigval2(1) << "  lambda2: " << eigval2(2) << endl;
    myfile << "In the two electron potential problem the difference between Jacobi's method and Armadillo is: " << eigval2(0) - A2.diag()(diagonal_A2(0)) << endl;
    myfile << "In the single electron potential problem the difference between Jacobi's method and Armadillo is: " << eigval1(0) - A1.diag()(diagonal_A1(0)) << endl;

    myfile.close();
}



//~~~~~~~~~~~~//
// Unit tests //
//~~~~~~~~~~~~//


void TestMaxElement(){
    int k,l, n = 3;
    double A_max;
    mat A = zeros<mat>(n,n);
    A.diag() += 1;
    A(0,2) = 1;
    A(0,2) = 3;
    A(1,2) = -100;
    A(1,0) = A(2,0) = A(2,1) = 10000; // findMax...-function searches only in upper triangular
    cout << "Testing the function for finding the max element "<< endl;
    cout << "A = " << endl;
    A.print();
    FindMaxElement(n, k, l, A, A_max);
    cout << "The max element was found to be: " << A(k,l) << endl;
    cout << "The max element should be: " << A(1,2) << endl;
}


void TestForEigenvalues(){

    int Test_n = 200;
    double Test_rho_max = 5.0;



    double Test_h = Test_rho_max / (Test_n+2);
    double Test_e = -1 / (Test_h*Test_h);
    double Test_d = 2 / (Test_h*Test_h);


    vec Test_rho(Test_n+2);

    for(int i=0; i<Test_n+2; i++){
        Test_rho(i) += Test_h*i;
    }
    Test_rho(0) = 0;


    vec Test_V1 = Test_rho % Test_rho;
    mat Test_A(Test_n,Test_n);

    for(int i=0; i<Test_n; i++){
        Test_A(i,i) = Test_d + Test_V1(i+1);
    }

    Test_A.diag(1) += Test_e;
    Test_A.diag(-1) += Test_e;
    int Test_iterations;
    uvec Test_diagonal_A;
    mat Test_U = eye<mat>(Test_n,Test_n); // matrix to contain eigenvectors


    cout << "Is our algorithm giving us the correct eigenvalues? " << endl;
    cout << "The results from the Jacobi algorithm " << endl;
    Jacobi(Test_n, Test_A, Test_diagonal_A, Test_U, Test_iterations);
    cout <<  "n: " << Test_n << endl;
    cout << "Number of iterations (Unit Test): " << Test_iterations << endl;
    cout << Test_diagonal_A(2) << endl;
    cout << "lambda0_t: " << Test_A.diag()(Test_diagonal_A(0)) << "  lambda1_t: " << Test_A.diag()(Test_diagonal_A(1)) << "  lambda2_t: " << Test_A.diag()(Test_diagonal_A(2)) << endl;
}
