<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class DataCSV extends Model
{
    use HasFactory;
    protected $table = "data_csv";
    public $timestamps = false;
    protected $guarded = [];
}


