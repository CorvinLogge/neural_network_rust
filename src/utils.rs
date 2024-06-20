use std::io;
use std::io::Cursor;
use std::num::ParseIntError;

use image::ImageError;
use rocket::{Request, Response};
use rocket::http::{ContentType, Status};
use rocket::response::Responder;

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

#[derive(Clone, Default)]
pub(crate) struct Error {
    message: String,
    status: Status,
}

impl Error {
    pub fn new(message: String, status: Status) -> Error {
        Error {
            message,
            status,
        }
    }

    pub fn back_prop() -> Error {
        Error {
            message: "Internal Server Error".to_string(),
            status: Status::InternalServerError,
        }
    }

    pub fn profile() -> Error {
        Error {
            message: "Failed profiling network".to_string(),
            status: Status::InternalServerError,
        }
    }

    pub fn io_error() -> Error {
        Error {
            message: "IO error".to_string(),
            status: Status::InternalServerError,
        }
    }
}

impl<'r> Responder<'r, 'static> for Error {
    fn respond_to(self, request: &'r Request<'_>) -> rocket::response::Result<'static> {
        let err_response = self.message;

        Response::build()
            .status(self.status)
            .header(ContentType::Text)
            .sized_body(err_response.len(), Cursor::new(err_response))
            .ok()
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::new(value.to_string(), Status::new(500))
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::new(value.to_string(), Status::BadRequest)
    }
}

impl From<ImageError> for Error {
    fn from(value: ImageError) -> Self {
        Self::new(value.to_string(), Status::new(500))
    }
}

impl From<ParseIntError> for Error {
    fn from(value: ParseIntError) -> Self {
        Self::new(value.to_string(), Status::new(500))
    }
}