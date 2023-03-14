
using System;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
//using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
namespace Benford
{
    public class BenfordsLaw
    {
    	public double[] arr;
    	public int[] distributions;
        MemoryBuffer1D<int, Stride1D.Dense> distBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> arrBuffer;
        

        MemoryBuffer1D<double, Stride1D.Dense> valsbuffer;
        



        Context context;
        Device dev;


        Accelerator accelerate;
    	public BenfordsLaw(double[] array)
        {
           	this.arr = array;
            this.distributions = new int[10];
            this.context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());

            this.dev = this.context.GetPreferredDevice(preferCPU: false);
            

        }
        public int[] getDist(){
        	int num;
            int n;
        	Console.WriteLine("Heree");
        	//this.arr = array;
        	for(int i = 0; i < this.arr.GetLength(0); i++){
        		num =  (int)this.arr[i];
                n = (int)this.arr[i];
        		while (num >= 10)
		        {
		            num = num / 10;

		        }
                if(num !=(int)(n/Math.Pow(10, (int)Math.Log10(n)))){
                    Console.Write(this.arr[i]);
                    Console.Write(", ");
                    Console.Write(num);
                    Console.Write(", ");
                    Console.Write((int)(n/Math.Pow(10, (int)Math.Log10(n))));
                    Console.Write(", ");
                    Console.WriteLine();
                }
                
		        this.distributions[num] += 1;
		        //print1d(this.distributions);
        	}
        	return this.distributions;

        }
        public double calcBenford(double x){
        	return Math.Log10(1+(1/x));
        }
        //R = n(∑xy) – (∑x)(∑y) / √[n∑x²-(∑x)²][n∑y²-(∑y)²
        private double linearCorrelation(){
		    //print1d(this.distributions);
        	//getDist();
        	double x = 0;
        	double y = 0;
        	double xy = 0;
        	double xsqr = 0;
        	double ysqr = 0;
        	double ben = 0.0;
        	double arrval = 0.0;
        	for(int i = 1; i < this.distributions.GetLength(0); i++){
        		ben = calcBenford(i);
        		//Console.Write("Ben: ");
        		//Console.WriteLine(ben);
        		arrval = (double)this.distributions[i]/this.arr.GetLength(0);
        		//Console.WriteLine(arrval);
        		x +=arrval;
        		y +=ben;
        		xy += (arrval *ben);
        		xsqr += (arrval * arrval);
        		ysqr += (ben * ben);
        	}
        	// Console.WriteLine("VALUES:");
        	// Console.WriteLine(x);
        	// Console.WriteLine(y);

        	// Console.WriteLine(xsqr);
        	// Console.WriteLine(ysqr);
        	// Console.WriteLine(xy);
        	// Console.WriteLine("_________________");
            
        	return ((9*xy) - (x*y))/Math.Sqrt(((9*xsqr) - (x*x))*((9*ysqr) - (y*y)));

        }
        private double linearCorrelationGPU(){
            accelerate = this.dev.CreateAccelerator(this.context);
            this.distBuffer = accelerate.Allocate1D<int>(new Index1D(10));
            this.valsbuffer = accelerate.Allocate1D<double>(new Index1D(5));

           



            this.arrBuffer =accelerate.Allocate1D<double>(new Index1D(this.arr.GetLength(0)));
            

            this.arrBuffer.CopyFromCPU(this.arr);

            var getDistKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                        
                ArrayView1D<int, Stride1D.Dense>>(getDistKernal); 
            var linearCorrKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                         
                ArrayView1D<double, Stride1D.Dense>,int>(linearCorrKernal); 
            var setBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueKernal);
            var setIntBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setIntBuffToValueKernal);
            setIntBuffToValueKern(this.distBuffer.Extent.ToIntIndex(), this.distBuffer.View, 0);
            setBuffToValueKern(this.valsbuffer.Extent.ToIntIndex(), this.valsbuffer.View, 0.0);
            

            getDistKern(this.arrBuffer.Extent.ToIntIndex(), this.arrBuffer.View,this.distBuffer.View);
            
            linearCorrKern(this.distBuffer.Extent.ToIntIndex(), this.distBuffer.View, this.valsbuffer.View,this.arr.GetLength(0));
           
            double[] valarr = this.valsbuffer.GetAsArray1D();
            double x= valarr[0];
            double y= valarr[1];

            double xsqr= valarr[2];
            double ysqr= valarr[3];

            double xy= valarr[4];

           
            return ((9*xy) - (x*y))/Math.Sqrt(((9*xsqr) - (x*x))*((9*ysqr) - (y*y)));


        }
        private double hybrid(){
            accelerate = this.dev.CreateAccelerator(this.context);
            this.distBuffer = accelerate.Allocate1D<int>(new Index1D(10));
            this.valsbuffer = accelerate.Allocate1D<double>(new Index1D(5));

           



            this.arrBuffer =accelerate.Allocate1D<double>(new Index1D(this.arr.GetLength(0)));
            

            this.arrBuffer.CopyFromCPU(this.arr);

            var getDistKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                        
                ArrayView1D<int, Stride1D.Dense>>(getDistKernal); 
            var linearCorrKern = accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                         
                ArrayView1D<double, Stride1D.Dense>,int>(linearCorrKernal); 
            var setBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueKernal);
            var setIntBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<int, Stride1D.Dense>,
                int>(
                setIntBuffToValueKernal);
            setIntBuffToValueKern(this.distBuffer.Extent.ToIntIndex(), this.distBuffer.View, 0);
            setBuffToValueKern(this.valsbuffer.Extent.ToIntIndex(), this.valsbuffer.View, 0.0);
            

            getDistKern(this.arrBuffer.Extent.ToIntIndex(), this.arrBuffer.View,this.distBuffer.View);
            this.distributions = this.distBuffer.GetAsArray1D();
            
            return linearCorrelation();
        }
        void print1d(int[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");

        }
        void print1d(double[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");

        }
        //(int)(n / (int)(Math.pow(10, digits)))
        static void getDistKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> totalView,
            

            ArrayView1D<int, Stride1D.Dense> distView
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            
            int num = (int)totalView[index];
            while (num >= 10)
            {
                num = num / 10;

            }
            
            Atomic.Add(ref distView[num], 1);

            
        }
        static void linearCorrKernal(Index1D index,
            ArrayView1D<int, Stride1D.Dense> distView,
            ArrayView1D<double, Stride1D.Dense> valView,

            int arrlength
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            if(index.X > 0){
                double arrval = (double)distView[index]/arrlength;
                double ben = Math.Log10(1+(1/(double)index.X));
                Atomic.Add(ref valView[0], arrval);
                Atomic.Add(ref valView[1], ben);

                Atomic.Add(ref valView[4], (arrval *ben));
                Atomic.Add(ref valView[2], (arrval * arrval));

                Atomic.Add(ref valView[3], (ben * ben));
                

            }
            
        }
        static void setBuffToValueKernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void setIntBuffToValueKernal(Index1D index, 
            ArrayView1D<int, Stride1D.Dense> buff, 
            int setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
        }
        static void Main(string[] args)
        {
        	Random rand = new Random();
        	double[] test= new double[1600000];//{423.1, 2435443.3, 64.3, 773.3, 4.35};
        	for(int i = 0; i < test.GetLength(0); i++){
        		test[i] = rand.Next(1, 100000);
        	}
        	BenfordsLaw x = new BenfordsLaw(test);
            //x.print1d(test);
            Console.WriteLine("BENFORD");
            Stopwatch stop = new Stopwatch();
            stop.Start();
            x.getDist();
        	Console.WriteLine(x.linearCorrelation());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);

            stop.Reset();
            stop.Start();

            Console.WriteLine(x.linearCorrelationGPU());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);

            stop.Reset();
            stop.Start();

            Console.WriteLine(x.hybrid());
            stop.Stop();
            Console.WriteLine(stop.ElapsedMilliseconds);

            //x.linearCorrelationGPU();
        	//int[] output = x.getDist();
        	// for(int i = 1; i < 10; i++){
        	// 	Console.WriteLine(x.calcBenford(i));
        	// }
        }
    }
}
