use image::ImageError;
use rocket::http::{ContentType, Status};
use rocket::response::Responder;
use rocket::serde::json::Json;
use rocket::{Request, Response};
use serde::Serialize;
use serde_json::json;
use std::fmt::{Display, Formatter};
use std::io;
use std::io::Cursor;
use std::num::ParseIntError;

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub enum ErrorKind {
    #[default]
    Other,
    IO,
    ParseInt,
    Image,
    Json,
    UnknownVariant,
    Profile,
    BackPropagation,
}

#[derive(Default, Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    status: Status,
    cause: Option<Box<dyn std::error::Error>>,
}

impl Error {
    pub fn new(
        kind: ErrorKind,
        message: String,
        status: Status,
        cause: Option<Box<dyn std::error::Error>>,
    ) -> Error {
        Error {
            kind,
            message,
            status,
            cause,
        }
    }

    pub fn back_prop() -> Error {
        Error {
            kind: ErrorKind::BackPropagation,
            message: "Error during backpropagation algorithm".to_string(),
            status: Status::InternalServerError,
            cause: None,
        }
    }

    pub fn profile() -> Error {
        Error {
            kind: ErrorKind::Profile,
            message: "Failed profiling network".to_string(),
            status: Status::InternalServerError,
            cause: None,
        }
    }

    pub fn io(message: String, cause: Option<Box<dyn std::error::Error>>) -> Error {
        Error {
            kind: ErrorKind::IO,
            message,
            status: Status::InternalServerError,
            cause,
        }
    }

    pub fn unknown_variant(expected: Vec<u8>, actual: u8) -> Error {
        Error {
            kind: ErrorKind::UnknownVariant,
            message: format!("Expected one of {expected:?} but found {actual}"),
            status: Status::InternalServerError,
            cause: None,
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause.as_deref()
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl<'r> Responder<'r, 'static> for Error {
    fn respond_to(self, request: &'r Request<'_>) -> rocket::response::Result<'static> {
        let mut error_trace: Vec<&dyn std::error::Error> = Vec::new();

        error_trace_recursive(&self, &mut error_trace);

        let mut err_response = format!("{self}");

        for error in error_trace.clone() {
            err_response.push_str(format!("\n\tCaused By: \n\t {}", error.to_string()).as_str());
        }

        let formatted_error_trace: Vec<String> =
            error_trace.iter().map(|error| format!("{error}")).collect();

        let body = Json::from(json!(
            {
                "status": self.status,
                "message": self.message,
                "type": self.kind,
                "error trace": formatted_error_trace
            }
        ));

        Response::build()
            .status(self.status)
            .header(ContentType::Text)
            .sized_body(None, Cursor::new(body.to_string()))
            .ok()
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::new(
            ErrorKind::IO,
            value.to_string(),
            Status::new(500),
            Some(Box::new(value)),
        )
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::new(
            ErrorKind::Json,
            value.to_string(),
            Status::BadRequest,
            Some(Box::new(value)),
        )
    }
}

impl From<ImageError> for Error {
    fn from(value: ImageError) -> Self {
        Self::new(
            ErrorKind::Image,
            value.to_string(),
            Status::new(500),
            Some(Box::new(value)),
        )
    }
}

impl From<ParseIntError> for Error {
    fn from(value: ParseIntError) -> Self {
        Self::new(
            ErrorKind::ParseInt,
            value.to_string(),
            Status::new(500),
            Some(Box::new(value)),
        )
    }
}

pub fn error_trace_recursive<'a>(
    error: &'a dyn std::error::Error,
    errors: &mut Vec<&'a dyn std::error::Error>,
) {
    if let Some(source) = error.source() {
        errors.push(source);
        error_trace_recursive(source, errors);
    }
}
