use rocket::http::Status;
use serde::{Deserialize, Serialize};

use crate::optimizers::Optimizer::{adam, rms_prop, sgd, sgd_m};
use crate::utils::Error;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Optimizer {
    sgd = 0,
    sgd_m = 1,
    rms_prop = 2,
    adam = 3,
}

impl Optimizer {
    pub fn from(u: u8) -> Result<Optimizer, Error> {
        match u {
            0 => { Ok(sgd) }
            1 => { Ok(sgd_m) }
            2 => { Ok(rms_prop) }
            3 => { Ok(adam) }
            _ => { Err(Error::new(format!("unknown variant '{}', expected one of '0', '1', '2', '3'", u), Status::BadRequest)) }
        }
    }
}