namespace CommonGeometry.Entity
{
    public abstract class GeometryData
    {
        public string type { get; set; }
    }

    public class GPoint3Data : GeometryData
    {
        public double x { get; set; }

        public double y { get; set; }

        public double z { get; set; }
    }

    public class GVector3Data : GeometryData
    {
        public double x { get; set; }

        public double y { get; set; }

        public double z { get; set; }
    }

    public class GVector4Data : GeometryData
    {
        public double x { get; set; }

        public double y { get; set; }

        public double z { get; set; }

        public double w { get; set; }
    }

    public class GMatrix4Data : GeometryData
    {
        public GVector4Data row1 { get; set; }

        public GVector4Data row2 { get; set; }

        public GVector4Data row3 { get; set; }

        public GVector4Data row4 { get; set; }
    }
}
