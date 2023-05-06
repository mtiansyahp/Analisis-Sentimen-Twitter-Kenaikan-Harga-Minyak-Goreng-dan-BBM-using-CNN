<?php

namespace App\Http\Controllers;

use App\Models\DataCSV;
use Illuminate\Http\Request;
use App\Http\Controllers\Controller;
use Illuminate\Support\Facades\Validator;
use Symfony\Component\HttpClient\HttpClient;
use Termwind\Components\Dd;

class HomepageController extends Controller
{
    //
    public function index(){
        return view('beranda');

    }
    // store logic
    // public function storeCsv(Request $request ){
    //     $validation = Validator::make(
    //         $request->all(),
    //         [
    //             // 'file-upload' => 'required|mimes:csv,txt|max:2048',
    //             'review' => 'required|string',
    //             // 'notelp' => 'required',

    //         ],
    //         [
    //             // 'file-upload.required' => 'Inputan File Harus Diisi',
    //             'review.required' => 'Inputan Review Harus Diisi',
    //             // 'notelp.required' => 'Inputan File Harus Diisi',
    //         ]
    //     );
    //     if ($validation->fails()) {
    //         return redirect()->back()->withErrors($validation->errors())->withInput();
    //     }
    //     $datacsv = new DataCSV();
    //     // $datacsv->data_csv = $request->file('file-upload')->store('csv', 'public');
    //     $datacsv->review = $request-> review ;
    //     // $datacsv->no_telp = $request-> notelp ;

    //     if($datacsv->save()){

    //         return redirect()->back()->with('success', 'Selamat Inputan Anda Berhasil Diisi!');
    //     }
    // }

    public function getData(Request $request ){
            // $gd= DataCSV::get()->all();
            // return view('beranda',compact('gd'));
            // how to make end point to API with port flask

            // nadia code
            $apiUrl ='http://127.0.0.1:5000/home';
            $postData = [
                'ulasan' => $request->review,
                // 'preproc' => $request->radioPreproc,
                // 'model' => $request->radioModel,
            ];
            $formData = http_build_query($postData);
            $headers = [
                'Content-Type' => 'application/x-www-form-urlencoded',
            ];

            // Send the POST request to the API and get the response
            $client = HttpClient::create();
            $response = $client->request('POST', $apiUrl, [
                'headers' => $headers,
                'body' => $formData,
            ]);
            // Get the response content
            // $content = json_decode($response->getContent());
            $content = $response->getContent()  ;
            $title = "Demo";
            dd($content);
            return view('beranda')->with(compact('content', 'title'));
            // return view('pages.demo', compact('contentJSON', 'title'));
            // return view('beranda',compact('gd'));

    }


    public function dokumentasi(){
        return view('dokumentasi');
    }

    // public function getDataAPI(){

    // }
}
