#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <string>
#include <fstream>

// Point.
class Point
{
public:
	double x, y;

	Point operator + (Point const &obj) {
		Point p;
		p.x = x + obj.x;
		p.y = y + obj.y;
		return p;
	}

	Point operator - (Point const &obj) {
		Point p;
		p.x = x - obj.x;
		p.y = y - obj.y;
		return p;
	}

	Point operator * (Point const &obj) {
		Point p;
		p.x = x * obj.x;
		p.y = y * obj.y;
		return p;
	}

	Point operator * (double const &obj) {
		Point p;
		p.x = x * obj;
		p.y = y * obj;
		return p;
	}

	bool operator == (Point const &obj) {
		bool result = false;
		if ((x == obj.x) && (y == obj.y))
		{
			result = true;
		}
		// else do nothing.
		return result;
	}
};

using namespace std;
using namespace std::chrono;

// Self Confidence.
const double C1 = 1.7;
// Swarm Confidence.
const double C2 = 2.2;
// Intertia Factor
const double W = 0.7;
// Particle Count.
const int ParticleCount = 250;
// Repeating Result Count.
const int MaxRepeat = 300;
// Max Iteration.
const int MaxIteration = 1000;

const long long RandPointCount = ParticleCount * MaxIteration;

enum FunctionTypeEnum{
    FunctionType_Booth = 0, FunctionType_Beale, FunctionType_Ackley
};

void GenerateRandPoint(Point input[RandPointCount]);

double BoothFunction(Point input);
double BealeFunction(Point input);
double AckleyFunction(Point input);
double ApplyFunction(FunctionTypeEnum functionType, Point input);

void GenerateInitialValues(FunctionTypeEnum functionType, Point X[ParticleCount], Point V[ParticleCount]);
double FindMinAndIndex(double* input, int size, int& minIndex);

void ReadPointFromFile(string fileName, Point *p);
void WritePointToFile(string fileName, Point *p, long long rowCount);
bool CheckPointFile(string fileName, long long rowCount);

int main()
{
	int threadCount = 8;

	omp_set_num_threads(threadCount);

	// Multiplicity "1".
  int minIndex = 0, gbestCount = 0;
	double minValue = 0.0;
	Point GBest, TempGBest;
	FunctionTypeEnum functionType = FunctionType_Ackley;

  // Multiplicity "MaxIteration".
	double FunctionResult[MaxIteration];

	// Multiplicity "ParticleCount".
	double Result[ParticleCount], oldValue[ParticleCount], newValue[ParticleCount];
	Point Vji[ParticleCount], Xji[ParticleCount], PBest[ParticleCount];

	Point C1RandPoint[RandPointCount], C2RandPoint[RandPointCount];

	bool isXFileExist = false, isVFileExist = false, isC1RandomFileExist = false, isC2RandomFileExist = false;

	isXFileExist        = CheckPointFile("Point.txt", ParticleCount);
	isVFileExist        = CheckPointFile("Speed.txt", ParticleCount);
	isC1RandomFileExist = CheckPointFile("C1.txt", RandPointCount);
	isC2RandomFileExist = CheckPointFile("C2.txt", RandPointCount);

    if (isXFileExist == false || isVFileExist == false || isC1RandomFileExist == false || isC2RandomFileExist == false)
    {
        cout << "Files are created\n";
        // Initial V and X with random number.
        GenerateInitialValues(functionType, Xji, Vji);
        GenerateRandPoint(C1RandPoint);
        GenerateRandPoint(C2RandPoint);

        WritePointToFile("Point.txt", Xji, ParticleCount);
        WritePointToFile("Speed.txt", Vji, ParticleCount);
        WritePointToFile("C1.txt", C1RandPoint, RandPointCount);
        WritePointToFile("C2.txt", C2RandPoint, RandPointCount);
    }
    else
    {
        cout << "Files are read\n";

        ReadPointFromFile("Point.txt", Xji);
        ReadPointFromFile("Speed.txt", Vji);
        ReadPointFromFile("C1.txt", C1RandPoint);
        ReadPointFromFile("C2.txt", C2RandPoint);
    }

	// Start timer.
	auto start = high_resolution_clock::now();

  //#pragma omp parallel for schedule(static, 1)
	#pragma omp parallel for num_threads(threadCount)
	for(int i = 0; i < ParticleCount; i++)
	{
        PBest[i] = Xji[i];

        // Apply function.
		    Result[i] = ApplyFunction(functionType, PBest[i]);
	}

	minValue = FindMinAndIndex(Result, ParticleCount, minIndex);

	GBest = PBest[minIndex];

	TempGBest.x = 0;
	TempGBest.y = 0;

	int lastIter = 0;
	for (int iter = 0; iter < MaxIteration && gbestCount < MaxRepeat; iter++)
	{
		// Apply formula.
		//#pragma omp parallel for schedule(static, 1)
		#pragma omp parallel for num_threads(threadCount)
		for (int i = 1;i < ParticleCount; i++)
		{
			  	Vji[i] = Vji[i] * W + (C1RandPoint[iter * ParticleCount + i - 1] * C1) * (PBest[i] - Xji[i]) + (C2RandPoint[iter * ParticleCount + i - 1] * C2) * (GBest - Xji[i]);

			  	Xji[i] = Xji[i] + Vji[i];

					oldValue[i] = ApplyFunction(functionType, PBest[i]);
					newValue[i] = ApplyFunction(functionType, Xji[i]);

					if (newValue[i] < oldValue[i])
					{
						PBest[i] = Xji[i];
					}
					// else do nothing.

					Result[i] = ApplyFunction(functionType, PBest[i]);
		}
		// End of the for loop.

		minValue = FindMinAndIndex(Result, ParticleCount, minIndex);

		GBest = PBest[minIndex];

		// for stop criterion
		if (TempGBest == GBest)
		{
			gbestCount = gbestCount + 1;
		}
		else
		{
			gbestCount = 0;
		}

		TempGBest = GBest;

		FunctionResult[iter] = ApplyFunction(functionType, GBest);

		lastIter = iter;
	}
	// end of the loop.

	// Stop timer and calculate differance.
	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

	cout <<"Thread Count: "<<threadCount<< " Iteration Count: "<< lastIter << " Global Min: "<<FunctionResult[lastIter - 1]<<endl;

	getchar();
}
// End of the main function.

double FindMinAndIndex(double *input, int size, int& minIndex)
{
	double min = input[0];

	for (int i = 0; i < size; i++)
	{
		if (min > input[i])
		{
			// min value.
			min = input[i];
			// min index.
			minIndex = i;
		}
		// else do nothing.
	}
	// end of the loop.

	return min;
}

void GenerateRandPoint(Point input[RandPointCount])
{
    for (long long i = 0; i < RandPointCount; i++)
    {
    	srand(time(NULL));
        input[i].x = double(rand() % 100) / 100;
        input[i].y = double(rand() % 100) / 100;
    }
    // end of the loop.
}

void GenerateInitialValues(FunctionTypeEnum functionType, Point X[ParticleCount], Point V[ParticleCount])
{
    int minXValue = 0, maxXValue = 0, minVValue = 0, maxVValue = 0;

    switch (functionType)
	{
		case FunctionType_Booth:
		{
			minXValue = -10;
			maxXValue = 10;
			minVValue = -20;
			maxVValue = 20;

			break;
		}
		case FunctionType_Beale:
		{
			minXValue = -4.5;
			maxXValue = 4.5;
            minVValue = -9;
			maxVValue = 9;
			break;
		}
		case FunctionType_Ackley:
		{
			minXValue = -32;
			maxXValue = 32;
            minVValue = -64;
			maxVValue = 64;
			break;
		}
		default:
		{
			break;
		}
	}

	for (int i = 0; i < ParticleCount; i++)
	{
        srand(clock());

	    X[i].x = minXValue + rand() % ( maxXValue - minXValue + 1 );
	    X[i].y = minXValue + rand() % ( maxXValue - minXValue + 1 );

	    V[i].x = minVValue + rand() % ( maxVValue - minVValue + 1 );
	    V[i].y = minVValue + rand() % ( maxVValue - minVValue + 1 );
	}
	// End of the loop.
}

double BoothFunction(Point input)
{
	double result = 0;

	result = pow( (input.x + 2 * input.y - 7), 2) + pow( (2 * input.x + input.y - 5), 2);

	return result;
}
double BealeFunction(Point input)
{
	double result = 0;

	result = pow( (1.5   - input.x + input.x * input.y), 2) +
		       pow( (2.25  - input.x + input.x * input.y * input.y), 2) +
		       pow( (2.625 - input.x + input.x * input.y * input.y * input.y), 2);

	return result;
}
double AckleyFunction(Point input)
{
	double result = 0;

	result = -200 * exp(-0.02 * sqrt(input.x * input.x + input.y * input.y) );

	return result;
}
double ApplyFunction(FunctionTypeEnum functionType, Point input)
{
	double result = 0.0;

	switch (functionType)
	{
		case FunctionType_Booth:
		{
			result = BoothFunction(input);
			break;
		}
		case FunctionType_Beale:
		{
			result = BealeFunction(input);
			break;
		}
		case FunctionType_Ackley:
		{
			result = AckleyFunction(input);
			break;
		}
		default:
		{

		}
	}
	// end of the switch.
	return result;
}
void ReadPointFromFile(string fileName, Point *p)
{
	ifstream infile;
	infile.open(fileName);
	// Else do nothing.

	long long index = 0, pointCount = 0;

	infile >>pointCount;

	while (!infile.eof())
	{
		infile >> p[index].x >> p[index].y;
		index++;
	}
	// End of the loop.
	infile.close();
}

void WritePointToFile(string fileName, Point *p, long long rowCount)
{
    ofstream outfile;
    outfile.open(fileName);

    // Write point count.
    outfile <<rowCount<<"\n";

    for (int i = 0; i < rowCount; i++)
    {
        outfile <<p[i].x<<" "<<p[i].y<<"\n";
    }
    // End of the loop.
    outfile.close();
}

bool CheckPointFile(string fileName, long long rowCount)
{
    bool result = false;

    ifstream infile;
	infile.open(fileName);

	long long pointCount = 0;

	infile >>pointCount;

	infile.close();

	if (pointCount == rowCount)
	{
        result = true;
	}
	// else do nothing.

	return result;
}
