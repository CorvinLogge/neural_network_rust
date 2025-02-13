use std::io;
use std::io::Cursor;
use std::num::ParseIntError;

use image::ImageError;
use rocket::http::{ContentType, Status};
use rocket::response::Responder;
use rocket::{Request, Response};

#[macro_export]
macro_rules! debug_only {
    ($expr:expr) => {
        let mut debug;

        unsafe { debug = DEBUG }

        if debug {
            $expr;
        }
    };

    ($block:block) => {
        let debug;

        unsafe { debug = DEBUG }

        if debug {
            $block;
        }
    };
}
