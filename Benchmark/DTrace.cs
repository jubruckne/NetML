namespace NetML.Benchmark;
using System;
using System.Diagnostics;
using System.Text.RegularExpressions;

public class CPUPerformanceMetrics
{
    public static (long branches, long mispredicts) GetBranchMetricsForProcess(int durationMs = 1000) {
        int pid = Process.GetCurrentProcess().Id;
        string script = $@"
            dtrace -n '
            profile:::profile-1001hz
            /arg0/
            {{
                @[""""branches""""] = sum(vtimestamp);
                @[""""mispredicts""""] = sum(arg0);
            }}
            tick-{durationMs}ms
            {{
                printa(@);
                exit(0);
            }}
            '
        ";

        // pid == {pid} &&

        ProcessStartInfo psi = new ProcessStartInfo
                               {
                                   FileName               = "/bin/bash",
                                   Arguments              = $"-c \"{script}\"",
                                   RedirectStandardOutput = true,
                                   UseShellExecute        = false,
                                   CreateNoWindow         = true,
                               };

        using (Process process = Process.Start(psi))
        {
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            long branches    = ParseDtraceOutput(output, "branches");
            long mispredicts = ParseDtraceOutput(output, "mispredicts");

            return (branches, mispredicts);
        }
    }

    private static long ParseDtraceOutput(string output, string metric) {
        Console.WriteLine(output.ToString());
        Match match = Regex.Match(output, $@"{metric}\s+(\d+)");
        Console.WriteLine(match.ToString());
        return match.Success ? long.Parse(match.Groups[1].Value) : 0;
    }
}