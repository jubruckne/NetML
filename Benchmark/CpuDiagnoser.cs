using System.Runtime.InteropServices;
using BenchmarkDotNet.Analysers;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Validators;

namespace NetML.Benchmark;

public class CpuDiagnoserAttribute : Attribute, IConfigSource {
	public IConfig Config { get; }

	public CpuDiagnoserAttribute()
		=> Config = ManualConfig.CreateEmpty().AddDiagnoser(new CpuDiagnoser());
}

public static class CPUInfo {
	[DllImport ("libc")]
	private static extern int sysctlbyname (string name, out int int_val, ref IntPtr length, IntPtr newp, IntPtr newlen);

	private static long get_sys_info(string param) {
		IntPtr size  = sizeof(int);
		var res = sysctlbyname(param, out var value, ref size, IntPtr.Zero, 0);
		return value;
	}

	public static int PhysicalCpuCount => (int)get_sys_info("hw.physicalcpu");
}

public class CpuDiagnoser : IDiagnoser {
	System.Diagnostics.Process proc;

	public CpuDiagnoser()
		=> this.proc = System.Diagnostics.Process.GetCurrentProcess();

	public IEnumerable<string> Ids => new[] { "CPU" };
	public IEnumerable<IExporter> Exporters => Array.Empty<IExporter>();
	public IEnumerable<IAnalyser> Analysers => Array.Empty<IAnalyser>();

	public void DisplayResults(ILogger logger) {}

	public RunMode GetRunMode(BenchmarkCase benchmarkCase)
		=> RunMode.NoOverhead;

	private long userStart, userEnd;
	private long privStart, privEnd;


	public void Handle(HostSignal signal, DiagnoserActionParameters parameters) {
		if(signal == HostSignal.BeforeActualRun) {
			userStart = proc.UserProcessorTime.Ticks;
			privStart = proc.PrivilegedProcessorTime.Ticks;
		} if(signal == HostSignal.AfterActualRun) {
			userEnd = proc.UserProcessorTime.Ticks;
			privEnd = proc.PrivilegedProcessorTime.Ticks;
		}
	}

	public IEnumerable<Metric> ProcessResults(DiagnoserResults results) {
		yield return new Metric(CpuUserMetricDescriptor.Instance, (userEnd - userStart) * 100d / results.TotalOperations);
		yield return new Metric(CpuPrivilegedMetricDescriptor.Instance, (privEnd - privStart) * 100d / results.TotalOperations);
	}

	public IEnumerable<ValidationError> Validate(ValidationParameters validationParameters) {
		yield break;
	}

	class CpuUserMetricDescriptor : IMetricDescriptor {
		internal static readonly IMetricDescriptor Instance = new CpuUserMetricDescriptor();

		public bool GetIsAvailable(Metric metric)
			=> true;

		public string Id => "CPU Usr Time";
		public string DisplayName => Id;
		public string Legend => Id;
		public string NumberFormat => "0.##";
		public UnitType UnitType => UnitType.Time;
		public string Unit => "ns";
		public bool TheGreaterTheBetter => false;
		public int PriorityInCategory => 1;
	}

	class CpuPrivilegedMetricDescriptor : IMetricDescriptor {
		internal static readonly IMetricDescriptor Instance = new CpuPrivilegedMetricDescriptor();

		public bool GetIsAvailable(Metric metric)
			=> true;

		public string Id => "CPU Sys Time";
		public string DisplayName => Id;
		public string Legend => Id;
		public string NumberFormat => "0.##";
		public UnitType UnitType => UnitType.Time;
		public string Unit => "ns";
		public bool TheGreaterTheBetter => false;
		public int PriorityInCategory => 1;
	}
}