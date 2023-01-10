using Autodesk.AutoCAD.Runtime;
using System.Collections.Generic;
using Newtonsoft.Json;
using THProject.Service.Service;
using THProject.THEntity.THEntity;
using Autodesk.AutoCAD.ApplicationServices;
using System.IO;

namespace TXYTest
{
    public class TestCommands
    {
        [CommandMethod("WALL2JSON", CommandFlags.UsePickSet)]
        public static void wall2json()
        {
            AFLAction act = new AFLAction();
            act.Run(new List<string>() { "读取选择集", "混合识别" });
            List<THWall> walls = new List<THWall>();
            act._PlaneFrame.frames.ForEach(frame =>
            {
                walls.AddRange(act._THObject.s_Walls[frame.WorkingName]);
            });
            Document doc = Application.DocumentManager.MdiActiveDocument;
            string doc_dir = System.IO.Path.GetDirectoryName(doc.Name);
            string output_dir = doc_dir + "\\output";
            if (!Directory.Exists(output_dir))
            {
                Directory.CreateDirectory(output_dir);
            }
            string objects_json = JsonConvert.SerializeObject(walls, Formatting.Indented);
            System.IO.File.WriteAllText(output_dir+"\\thwalls.json", objects_json);
        }
    }
}
