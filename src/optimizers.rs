use rocket::http::Status;
use serde::{Deserialize, Serialize};

use crate::optimizers::Optimizer::{Adam, RmsProp, Sgd, SgdM};
use crate::utils::Error;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Optimizer {
    #[serde(alias = "sgd")] Sgd = 0,
    #[serde(alias = "sgdm")] SgdM = 1,
    #[serde(alias = "rms_prop")] RmsProp = 2,
    #[serde(alias = "adam")] Adam = 3,
}

impl Optimizer {
    pub fn from(u: u8) -> Result<Optimizer, Error> {
        match u {
            0 => { Ok(Sgd) }
            1 => { Ok(SgdM) }
            2 => { Ok(RmsProp) }
            3 => { Ok(Adam) }
            _ => { Err(Error::new(format!("unknown variant '{}', expected one of '0', '1', '2', '3'", u), Status::BadRequest)) }
        }
    }
}