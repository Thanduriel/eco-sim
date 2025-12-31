use bevy::color::{LinearRgba};
use bevy::prelude::StableInterpolate;

const INCANDESCENT: [LinearRgba; 11] = [
    LinearRgba::rgb(0.807843137254902, 1.0, 1.0),
    LinearRgba::rgb(0.7764705882352941, 0.9686274509803922, 0.8392156862745098),
    LinearRgba::rgb(0.6352941176470588, 0.9568627450980393, 0.6078431372549019),
    LinearRgba::rgb(0.7333333333333333, 0.8941176470588236, 0.3254901960784314),
    LinearRgba::rgb(0.8352941176470589, 0.807843137254902, 0.01568627450980392),
    LinearRgba::rgb(0.9058823529411765, 0.7098039215686275, 0.011764705882352941),
    LinearRgba::rgb(0.9450980392156862, 0.6, 0.011764705882352941),
    LinearRgba::rgb(0.9647058823529412, 0.4745098039215686, 0.043137254901960784),
    LinearRgba::rgb(0.9764705882352941, 0.28627450980392155, 0.00784313725490196),
    LinearRgba::rgb(0.8941176470588236, 0.0196078431372549, 0.08235294117647059),
    LinearRgba::rgb(0.6588235294117647, 0.0, 0.011764705882352941),
];
const INCANDESCENT_BAD_DATA: LinearRgba =
    LinearRgba::rgb(0.5333333333333333, 0.5333333333333333, 0.5333333333333333);

const RAINBOW: [LinearRgba; 34] = [
    LinearRgba::rgb(0.9098039215686274, 0.9254901960784314, 0.984313725490196),
    LinearRgba::rgb(0.8666666666666667, 0.8470588235294118, 0.9372549019607843),
    LinearRgba::rgb(0.8196078431372549, 0.7568627450980392, 0.8823529411764706),
    LinearRgba::rgb(0.7647058823529411, 0.6588235294117647, 0.8196078431372549),
    LinearRgba::rgb(0.7098039215686275, 0.5607843137254902, 0.7607843137254902),
    LinearRgba::rgb(0.6549019607843137, 0.47058823529411764, 0.7058823529411765),
    LinearRgba::rgb(0.6078431372549019, 0.3843137254901961, 0.6549019607843137),
    LinearRgba::rgb(0.5490196078431373, 0.3058823529411765, 0.6),
    LinearRgba::rgb(0.43529411764705883, 0.2980392156862745, 0.6078431372549019),
    LinearRgba::rgb(0.3764705882352941, 0.34901960784313724, 0.6627450980392157),
    LinearRgba::rgb(0.3333333333333333, 0.40784313725490196, 0.7215686274509804),
    LinearRgba::rgb(0.3058823529411765, 0.4745098039215686, 0.7725490196078432),
    LinearRgba::rgb(0.30196078431372547, 0.5411764705882353, 0.7764705882352941),
    LinearRgba::rgb(0.3058823529411765, 0.5882352941176471, 0.7372549019607844),
    LinearRgba::rgb(0.32941176470588235, 0.6196078431372549, 0.7019607843137254),
    LinearRgba::rgb(0.34901960784313724, 0.6470588235294118, 0.6627450980392157),
    LinearRgba::rgb(0.3764705882352941, 0.6705882352941176, 0.6196078431372549),
    LinearRgba::rgb(0.4117647058823529, 0.6941176470588235, 0.5647058823529412),
    LinearRgba::rgb(0.4666666666666667, 0.7176470588235294, 0.49019607843137253),
    LinearRgba::rgb(0.5490196078431373, 0.7372549019607844, 0.40784313725490196),
    LinearRgba::rgb(0.6509803921568628, 0.7450980392156863, 0.32941176470588235),
    LinearRgba::rgb(0.7450980392156863, 0.7372549019607844, 0.2823529411764706),
    LinearRgba::rgb(0.8196078431372549, 0.7098039215686275, 0.2549019607843137),
    LinearRgba::rgb(0.8666666666666667, 0.6666666666666666, 0.23529411764705882),
    LinearRgba::rgb(0.8941176470588236, 0.611764705882353, 0.2235294117647059),
    LinearRgba::rgb(0.9058823529411765, 0.5490196078431373, 0.20784313725490197),
    LinearRgba::rgb(0.9019607843137255, 0.4745098039215686, 0.19607843137254902),
    LinearRgba::rgb(0.8941176470588236, 0.38823529411764707, 0.17647058823529413),
    LinearRgba::rgb(0.8745098039215686, 0.2823529411764706, 0.1568627450980392),
    LinearRgba::rgb(0.8549019607843137, 0.13333333333333333, 0.13333333333333333),
    LinearRgba::rgb(0.7215686274509804, 0.13333333333333333, 0.11764705882352941),
    LinearRgba::rgb(0.5843137254901961, 0.12941176470588237, 0.10588235294117647),
    LinearRgba::rgb(0.4470588235294118, 0.11764705882352941, 0.09019607843137255),
    LinearRgba::rgb(0.3215686274509804, 0.10196078431372549, 0.07450980392156863),
];

const RAINBOW_BAD_DATA: LinearRgba = LinearRgba::rgb(0.4, 0.4, 0.4);

#[allow(dead_code)]
pub enum ColorScheme {
    Incandescent,
    Rainbow,
}

fn get_color_scheme(color_scheme: ColorScheme) -> (&'static [LinearRgba], &'static LinearRgba) {
    match color_scheme {
        ColorScheme::Incandescent => (&INCANDESCENT, &INCANDESCENT_BAD_DATA),
        ColorScheme::Rainbow => (&RAINBOW, &RAINBOW_BAD_DATA),
    }
}

pub struct ColorMap {
    min: f32,
    max: f32,
    colors: Vec<LinearRgba>,
    bad_data_color: LinearRgba,
}

impl ColorMap {
    pub fn new(min: f32, max: f32, color_scheme: ColorScheme) -> Self {
		let (colors, bad_color) = get_color_scheme(color_scheme);
        ColorMap {
            min: min,
            max: max,
            colors: colors.into(),
            bad_data_color: *bad_color,
        }
    }

    pub fn get_color(&self, value: f32) -> LinearRgba {
        // nan or infinite: bad data
        if !value.is_finite() {
            return self.bad_data_color;
        }

        if self.min == self.max {
            return self.colors[0];
        }

        // scale value to [0,1]
        let v_scaled = ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
        //let v_scaled = (value.clamp(self.min, self.max) - self.min) / self.max;
        let v_select = v_scaled * (self.colors.len() - 1) as f32;
        let lower_idx = v_select.trunc() as usize;
        let upper_idx = std::cmp::min(lower_idx + 1, self.colors.len() - 1);
        let t = v_select.fract();

        self.colors[lower_idx].interpolate_stable(&self.colors[upper_idx], t)
    }
}
