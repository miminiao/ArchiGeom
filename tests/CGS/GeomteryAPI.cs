using CommonGeometry.Entity;
using CommonGeometry.Service.Extensions;
using CommonGeometry.Utils;
using System.Threading.Tasks;

namespace CommonGeometry.Service.Utils
{
    /// <exclude />
    public class GeometryAPI: IGeometryAPI
    {
        public async Task<GPoint3> Point3AddVector3(GPoint3 point, GVector3 vector)
        {
            return point + vector;
        }

        public async Task<GPoint3> Point3SubtractVector3(GPoint3 point, GVector3 vector)
        {
            return point - vector;
        }

        public async Task<GVector3> Point3DisplacementToPoint3(GPoint3 point1, GPoint3 point2)
        {
            return point2 - point1;
        }

        public async Task<GPoint3> Point3MultiplyMatrix4(GPoint3 point, GMatrix4 matrix)
        {
            return point.Multiplied(matrix);
        }

        public async Task<double> Point3DistanceToPoint3(GPoint3 point1, GPoint3 point2)
        {
            return point1.DistanceTo(point2);
        }

        public async Task<GPoint3> Point3MidToPoint3(GPoint3 point1, GPoint3 point2)
        {
            return point1.GetMid(point2);
        }

        public async Task<double> Vector3DotToVector3(GVector3 vector1, GVector3 vector2)
        {
            return vector1.Dot(vector2);
        }

        public async Task<GVector3> Vector3CrossToVector3(GVector3 vector1, GVector3 vector2)
        {
            return vector1.Cross(vector2);
        }

        public async Task<GVector3> Vector3MultiplyMatrix4(GVector3 vector, GMatrix4 matrix)
        {
            return vector.Multiplied(matrix);
        }

        public async Task<GVector3> Vector3RotateByAngle(GVector3 vector, double angle, GVector3 rotationAxis)
        {
            return vector.Rotated(angle, rotationAxis);
        }

        public async Task<double> Vector3AngleToVector3(GVector3 vector1, GVector3 vector2)
        {
            return vector1.AngleTo(vector2);
        }
        public async Task<double> Vector3AngleToVector3(GVector3 vector1, GVector3 vector2, GVector3 referenceVec)
        {
            return vector1.AngleTo(vector2, referenceVec);
        }
    }

    /// <exclude />
    public interface IGeometryAPI
    {
        Task<GPoint3> Point3AddVector3(GPoint3 point, GVector3 vector);
        Task<GPoint3> Point3SubtractVector3(GPoint3 point, GVector3 vector);
        Task<GVector3> Point3DisplacementToPoint3(GPoint3 point1, GPoint3 point2);
        Task<GPoint3> Point3MultiplyMatrix4(GPoint3 point1, GMatrix4 matrix);
        Task<double> Point3DistanceToPoint3(GPoint3 point1, GPoint3 point2);
        Task<GPoint3> Point3MidToPoint3(GPoint3 point1, GPoint3 point2);
        Task<double> Vector3DotToVector3(GVector3 vector1, GVector3 vector2);
        Task<GVector3> Vector3CrossToVector3(GVector3 vector1, GVector3 vector2);
        Task<GVector3> Vector3MultiplyMatrix4(GVector3 vector, GMatrix4 matrix);
        Task<GVector3> Vector3RotateByAngle(GVector3 vector, double angle, GVector3 rotationAxis);
        Task<double> Vector3AngleToVector3(GVector3 vector1, GVector3 vector2);
        Task<double> Vector3AngleToVector3(GVector3 vector1, GVector3 vector2, GVector3 referenceVec);

    }
}
