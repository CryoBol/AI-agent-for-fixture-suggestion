import cadquery as cq
import numpy as np

def generate_design1_assembly(w_dia, d_dia, lg, hole_count):
    """
    Design 1: Original Standard Stack
    Features: Distinct top/bottom plates, central ceramic hourglass core, and alignment pins.
    """
    # 1. Base Stand
    base = (cq.Workplane("XY")
            .circle(d_dia/2 + 20).extrude(8)
            .faces(">Z").workplane()
            .circle(d_dia/2 + 10).extrude(4))

    # 2. Retention Plate (Used for Top and Bottom)
    plate = (cq.Workplane("XY")
             .circle(d_dia/2 + 10).extrude(6)
             .faces(">Z").workplane()
             .polarArray(d_dia/2 - 2, 0, 360, hole_count).circle(0.5).cutThruAll() # Nitinol holes
             .faces(">Z").workplane()
             .polarArray(d_dia/2 + 6, 0, 360, 4).circle(1.6).cutThruAll() # Pin holes
             .faces(">Z").workplane()
             .circle(4.0).cutThruAll()) # Central bore

    # 3. Precision Ceramic Waist Core (Corrected Revolve)
    core = (cq.Workplane("XZ")
            .moveTo(0, 0)
            .lineTo(d_dia/2 - 2, 0)
            .lineTo(d_dia/2 - 2, lg * 0.15)
            .lineTo(w_dia/2, lg * 0.5)  # The pinch
            .lineTo(d_dia/2 - 2, lg * 0.85)
            .lineTo(d_dia/2 - 2, lg)
            .lineTo(0, lg)
            .close()
            .revolve(360, (0,0,0), (0,0,1))
            .faces(">Z").workplane().circle(4.0).cutThruAll())

    # 4. Alignment Pin
    pin = cq.Workplane("XY").circle(1.5).extrude(lg + 16)

    # --- Assembly Stacking ---
    assy = cq.Assembly(name="Design1_Original")
    assy.add(base, name="Base", color=cq.Color("darkgray"))
    
    z_base = 12
    z_core = z_base + 6
    z_top = z_core + lg
    
    assy.add(plate, name="BottomPlate", loc=cq.Location(cq.Vector(0, 0, z_base)), color=cq.Color("lightgray"))
    assy.add(core, name="CeramicCore", loc=cq.Location(cq.Vector(0, 0, z_core)), color=cq.Color("wheat"))
    assy.add(plate, name="TopPlate", loc=cq.Location(cq.Vector(0, 0, z_top)), color=cq.Color("lightgray"))
    
    for i in range(4):
        angle = np.radians(i * 90)
        rad = d_dia/2 + 6
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        assy.add(pin, name=f"Pin_{i}", loc=cq.Location(cq.Vector(x, y, z_base)), color=cq.Color("silver"))

    return assy


def generate_design2_assembly(w_dia, d_dia, lg, hole_count):
    """
    Design 2: Integrated Core
    Features: Consolidates the retention plate and waist core into two self-aligning halves.
    """
    half_lg = lg / 2

    def make_half():
        # Integrated base and core half
        return (cq.Workplane("XY")
                .circle(d_dia/2 + 10).extrude(6)
                .faces(">Z").workplane()
                .circle(d_dia/2).extrude(half_lg)
                # Fillet to create the hourglass pinch natively
                .edges(">Z and %CIRCLE").fillet(w_dia/4) 
                .faces(">Z").workplane()
                .polarArray(d_dia/2 - 2, 0, 360, hole_count).circle(0.5).cutThruAll()
                .faces(">Z").workplane()
                .circle(4.0).cutThruAll())

    bottom_half = make_half()
    
    # Flip the top half to mirror the bottom
    top_half = make_half().rotate((0,0,0), (1,0,0), 180)

    assy = cq.Assembly(name="Design2_Integrated")
    assy.add(bottom_half, name="BottomHalf", loc=cq.Location(cq.Vector(0, 0, 0)), color=cq.Color("lightgray"))
    assy.add(top_half, name="TopHalf", loc=cq.Location(cq.Vector(0, 0, 12 + lg)), color=cq.Color("gray"))

    return assy


def generate_design3_assembly(w_dia, d_dia, lg, hole_count):
    """
    Design 3: Radial Tensioning
    Features: A massive outer compression ring with radial tensioning clamps.
    """
    # 1. Central Hub
    hub = (cq.Workplane("XY")
           .circle(d_dia/2).extrude(lg + 12)
           .faces(">Z").workplane()
           .polarArray(d_dia/2 - 3, 0, 360, hole_count).circle(0.5).cutThruAll())

    # 2. Outer Compression Ring
    ring_outer_rad = d_dia/2 + 25
    ring_inner_rad = d_dia/2 + 5
    
    ring = (cq.Workplane("XY")
            .circle(ring_outer_rad).circle(ring_inner_rad).extrude(lg + 12))
    
    # Drill radial holes through the side of the ring
    for i in range(8):
        angle = i * 45
        ring = (ring.faces(">Z").workplane()
                .transformed(offset=cq.Vector(0, 0, -(lg/2 + 6)), rotate=cq.Vector(90, angle, 0))
                .circle(2.0).cutThruAll())

    # 3. Tensioning Screws (Pins)
    screw = cq.Workplane("XY").circle(1.8).extrude(20)

    assy = cq.Assembly(name="Design3_Radial")
    assy.add(hub, name="CenterHub", color=cq.Color("lightgray"))
    assy.add(ring, name="OuterRing", color=cq.Color("darkgray"))
    
    for i in range(8):
        angle = np.radians(i * 45)
        # Position screws radially pointing inward
        x = (ring_outer_rad - 5) * np.cos(angle)
        y = (ring_outer_rad - 5) * np.sin(angle)
        # Rotate the screw to align with the radial axis
        loc = cq.Location(cq.Vector(x, y, (lg+12)/2), cq.Vector(0, 0, 1), np.degrees(angle) + 90)
        # We rotate it 90 degrees on X to lay it flat, then rotate by 'angle' around Z
        loc = cq.Location(cq.Vector(x, y, (lg+12)/2)) * cq.Location(cq.Vector(0,0,0), cq.Vector(0,0,1), i*45) * cq.Location(cq.Vector(0,0,0), cq.Vector(1,0,0), 90)
        
        assy.add(screw, name=f"Screw_{i}", loc=loc, color=cq.Color("silver"))

    return assy